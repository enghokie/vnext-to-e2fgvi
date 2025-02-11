# -*- coding: utf-8 -*-
import cv2
from PIL import Image
import numpy as np
import importlib
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
import torch
from datetime import datetime
import random

from core.utils import to_tensors

# VNext imports
from VNext.detectron2.config import get_cfg
from VNext.detectron2.engine.defaults import DefaultPredictor
from VNext.detectron2.utils.visualizer import ColorMode, Visualizer


# Args
parser = argparse.ArgumentParser(description="E2FGVI")
parser.add_argument("-v", "--video", type=str)
parser.add_argument("-c", "--ckpt", type=str, required=True)
parser.add_argument("--model", type=str, choices=['e2fgvi', 'e2fgvi_hq'])
parser.add_argument("--step", type=int, default=10)
parser.add_argument("--num_ref", type=int, default=-1)
parser.add_argument("--neighbor_stride", type=int, default=5)
parser.add_argument("--savefps", type=int, default=24)

# Args for e2fgvi_hq (which can handle videos with arbitrary resolution)
parser.add_argument("--set_size", action='store_true', default=False)
parser.add_argument("--width", type=int)
parser.add_argument("--height", type=int)

# VNext args
parser.add_argument("--confidence-threshold", type=float, default=0.5)

# Webcam function args 
parser.add_argument("--webcam", type=bool, default=False)
parser.add_argument("--max_frames", type=int, default=50)

args = parser.parse_args()


# E2FGVI constants
ref_length = args.step  # ref_step
num_ref = args.num_ref
neighbor_stride = args.neighbor_stride
default_fps = args.savefps
CLASS_PERSON_VALUE = 0

# VNext constants
VNEXT_CONFIG_PATH = "VNext/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(VNEXT_CONFIG_PATH)
    cfg.merge_from_list([])
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


# sample reference frames from the whole video
def get_ref_index(f, neighbor_ids, length):
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref // 2))
        end_idx = min(length, f + ref_length * (num_ref // 2))
        for i in range(start_idx, end_idx + 1, ref_length):
            if i not in neighbor_ids:
                if len(ref_index) > num_ref or i > (length - 1):
                    break
                ref_index.append(i)
    return ref_index

# sample references from video history (used with webcam version)
def get_live_ref_index(f, neighbor_ids, length):
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * num_ref)
        end_idx = length
        for i in range(start_idx, end_idx + 1, ref_length):
            if i not in neighbor_ids:
                if len(ref_index) > num_ref:
                    break 
                ref_index.append(i)
    return ref_index 

def preprocess_masks(masks, size):
    out_masks = []
    for m in masks:
        m = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m,
                       cv2.getStructuringElement(cv2.MORPH_CROSS, (6, 6)),
                       iterations=4)
        out_masks.append(Image.fromarray(m * 255))
    
    return out_masks


#  read frames from video
def read_frame_from_videos(args):
    vname = args.video
    frames = []
    if args.use_mp4:
        vidcap = cv2.VideoCapture(vname)
        success, image = vidcap.read()
        count = 0
        while success:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image)
            success, image = vidcap.read()
            count += 1
    else:
        lst = os.listdir(vname)
        lst.sort()
        fr_lst = [vname + '/' + name for name in lst]
        for fr in fr_lst:
            image = cv2.imread(fr)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image)
    return frames


# resize frames
def resize_frames(frames, size=None):
    if size is not None:
        frames = [f.resize(size) for f in frames]
    else:
        size = frames[0].size
    return frames, size


def main_video():
    # Set up VNext predictor
    vnext_cfg = setup_cfg(args)
    vnext_seg_predictor = DefaultPredictor(vnext_cfg)

    # set up models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "e2fgvi":
        size = (432, 240)
    elif args.set_size:
        size = (args.width, args.height)
    else:
        size = None

    net = importlib.import_module('model.' + args.model)
    model = net.InpaintGenerator().to(device)
    data = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(data)
    print(f'Loading model from: {args.ckpt}')
    model.eval()

    # prepare datset
    args.use_mp4 = True if args.video.endswith('.mp4') else False
    print(
        f'Loading videos from: {args.video} | INPUT MP4 format: {args.use_mp4}'
    )
    frames = read_frame_from_videos(args)
    frames, size = resize_frames(frames, size)
    h, w = size[1], size[0]
    video_length = len(frames)
    imgs = to_tensors()(frames).unsqueeze(0) * 2 - 1
    frames = [np.array(f).astype(np.uint8) for f in frames]
    comp_frames = [None] * video_length

    # prepare VNext
    vnext_viz_colormode = ColorMode.IMAGE_BW

    # completing holes by e2fgvi
    print(f'Start test...')
    sequences = 0
    total_frames = 0
    vnext_model_elapsed = 0.0
    vnext_process_elapsed = 0.0
    e2fgvi_model_elapsed = 0.0
    e2fgvi_process_elapsed = 0.0
    total_vnext_model_elapsed = 0.0
    total_vnext_process_elapsed = 0.0
    total_e2fgvi_model_elapsed = 0.0
    total_e2fgvi_process_elapsed = 0.0
    for f in tqdm(range(0, video_length, neighbor_stride)):
        sequences += 1
        neighbor_ids = [
            i for i in range(max(0, f - neighbor_stride),
                                min(video_length, f + neighbor_stride + 1))
        ]
        ref_ids = get_ref_index(f, neighbor_ids, video_length)
        all_ids = neighbor_ids + ref_ids
        total_frames += len(all_ids)

        print(f'Neighbor frame IDs: {neighbor_ids}')
        print(f'Reference frame IDs: {ref_ids}')

        # Get the mask images from VNext predictor
        vnext_masks = []
        vnext_process_start = datetime.now()
        for idx in neighbor_ids + ref_ids:
            vnext_model_start = datetime.now()
            predictions = vnext_seg_predictor(frames[idx])
            vnext_model_elapsed += (datetime.now() - vnext_model_start).total_seconds()

            # If we don't have any detections then just use the original frame
            if 'instances' not in predictions or len(predictions['instances']) <= 0:
                vnext_masks.append(frames[idx])
                continue

            # Only use segmentation masks that are not of people (remove occlusions for people)
            instances = predictions['instances'].to(torch.device("cpu"))
            instances = instances[instances.pred_classes != CLASS_PERSON_VALUE]

            # Remove our boxes, classes, and scores from the segmentation instances before
            # we draw our masked images to prevent the labels and boxes from being drawn
            pred_boxes = instances.get('pred_boxes').tensor.numpy() if instances.has('pred_boxes') else None
            pred_scores = instances.get('scores') if instances.has('scores') else None
            pred_classes = instances.pred_classes.tolist() if instances.has("pred_classes") else None

            if instances.has('pred_boxes'):
                instances.remove('pred_boxes')
            if instances.has('scores'):
                instances.remove('scores')
            if instances.has('pred_classes'):
                instances.remove('pred_classes')

            # Get our segmentation masked images
            masked_img = np.zeros_like(frames[idx])
            masked_img = masked_img[:, :, ::-1]
            visualizer = Visualizer(masked_img, None, instance_mode=vnext_viz_colormode)
            vis_frame = visualizer.draw_instance_predictions(predictions=instances)
            vnext_masks.append(cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR))

        masks = preprocess_masks(vnext_masks, size)
        binary_masks = [
            np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks
        ]

        '''# Visualize the masked images
        for idx in range(len(tmp)):
            m = np.array(masks[idx])
            cv2.namedWindow("Mask img: " + str(tmp[idx]), cv2.WINDOW_NORMAL)
            cv2.imshow("Mask img: " + str(tmp[idx]), m)
            if cv2.waitKey(0) == 27:
                break
        '''

        masks = to_tensors()(masks).unsqueeze(0)
        vnext_process_elapsed += (datetime.now() - vnext_process_start).total_seconds()
        e2fgvi_process_start = datetime.now()
        selected_imgs = imgs[:1, all_ids, :, :, :]
        selected_imgs, masks = selected_imgs.to(device), masks.to(device)
        with torch.no_grad():
            masked_imgs = selected_imgs * (1 - masks)
            '''# Visualize original images with mask holes
            for idx in range(len(all_ids)):
                m = np.array(masked_imgs.to(torch.device('cpu'))).squeeze()[idx].transpose(1, 2, 0) * 255
                m = cv2.cvtColor(m.astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.namedWindow("Mask img: " + str(all_ids[idx]), cv2.WINDOW_NORMAL)
                cv2.imshow("Mask img: " + str(all_ids[idx]), m)
                if cv2.waitKey(0) == 27:
                    break
            '''

            mod_size_h = 60
            mod_size_w = 108
            h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
            w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
            masked_imgs = torch.cat(
                [masked_imgs, torch.flip(masked_imgs, [3])],
                3)[:, :, :, :h + h_pad, :]
            masked_imgs = torch.cat(
                [masked_imgs, torch.flip(masked_imgs, [4])],
                4)[:, :, :, :, :w + w_pad]

            e2fgvi_model_start = datetime.now()
            pred_imgs, _ = model(masked_imgs, len(neighbor_ids))
            e2fgvi_model_elapsed += (datetime.now() - e2fgvi_model_start).total_seconds()
            pred_imgs = pred_imgs[:, :, :h, :w]
            pred_imgs = (pred_imgs + 1) / 2
            pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_imgs[i]).astype(
                    np.uint8) * binary_masks[i] + frames[idx] * (
                        1 - binary_masks[i])

                '''# Visualize the intermediate results
                bm = binary_masks[i]
                bm[bm > 0] = 255
                cv2.namedWindow("binary mask: " + str(idx), cv2.WINDOW_NORMAL)
                cv2.imshow("binary mask: " + str(idx), bm)
                cv2.namedWindow("frame: " + str(idx), cv2.WINDOW_NORMAL)
                cv2.imshow("frame: " + str(idx), frames[idx])
                cv2.namedWindow("pred img: " + str(idx), cv2.WINDOW_NORMAL)
                cv2.imshow("pred img: " + str(idx), np.array(pred_imgs[i]).astype(np.uint8))
                cv2.namedWindow("img: " + str(idx), cv2.WINDOW_NORMAL)
                cv2.imshow("img: " + str(idx), img)
                if cv2.waitKey(0) == 27:
                    break
                '''

                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(
                        np.float32) * 0.5 + img.astype(np.float32) * 0.5

            e2fgvi_process_elapsed += (datetime.now() - e2fgvi_process_start).total_seconds()
            if sequences == 10:
                total_vnext_model_elapsed += vnext_model_elapsed / 10
                total_vnext_process_elapsed += vnext_process_elapsed / 10
                total_e2fgvi_model_elapsed += e2fgvi_model_elapsed / 10
                total_e2fgvi_process_elapsed += e2fgvi_process_elapsed / 10

                vnext_model_elapsed = 0
                vnext_process_elapsed = 0
                e2fgvi_model_elapsed = 0
                e2fgvi_process_elapsed = 0

    # log time results
    if sequences % 10 != 0:
        total_vnext_model_elapsed += vnext_model_elapsed / sequences
        total_vnext_process_elapsed += vnext_process_elapsed / sequences
        total_e2fgvi_model_elapsed += e2fgvi_model_elapsed / sequences
        total_e2fgvi_process_elapsed += e2fgvi_process_elapsed / sequences

    print(f'Total sequences: {sequences}, total frames: {total_frames}')
    print(f'VNext model time avg: {total_vnext_model_elapsed * 1000}ms per sequence')
    print(f'VNext process time avg: {total_vnext_process_elapsed * 1000}ms per sequence')
    print(f'E2FGVI model time avg: {total_e2fgvi_model_elapsed * 1000}ms per sequence')
    print(f'E2FGVI process time avg: {total_e2fgvi_process_elapsed * 1000}ms per sequence')
    print(f'VNext model time avg: {(total_vnext_model_elapsed * sequences * 1000) / total_frames}ms per frame')
    print(f'VNext process time avg: {(total_vnext_process_elapsed * sequences * 1000) / total_frames}ms per frame')
    print(f'E2FGVI model time avg: {(total_e2fgvi_model_elapsed * sequences * 1000) / total_frames}ms per frame')
    print(f'E2FGVI process time avg: {(total_e2fgvi_process_elapsed * sequences * 1000) / total_frames}ms per frame')

    # saving videos
    print('Saving videos...')
    save_dir_name = 'results'
    ext_name = '_results.mp4'
    save_base_name = args.video.split('/')[-1]
    save_name = save_base_name.replace(
        '.mp4', ext_name) if args.use_mp4 else save_base_name + ext_name
    if not os.path.exists(save_dir_name):
        os.makedirs(save_dir_name)
    save_path = os.path.join(save_dir_name, save_name)
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                default_fps, size)
    for f in range(video_length):
        comp = comp_frames[f].astype(np.uint8)
        writer.write(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
    writer.release()
    print(f'Finish test! The result video is saved in: {save_path}.')

    # show results
    print('Let us enjoy the result!')
    fig = plt.figure('Let us enjoy the result')
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.axis('off')
    ax1.set_title('Original Video')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis('off')
    ax2.set_title('Our Result')
    imdata1 = ax1.imshow(frames[0])
    imdata2 = ax2.imshow(comp_frames[0].astype(np.uint8))

    def update(idx):
        imdata1.set_data(frames[idx])
        imdata2.set_data(comp_frames[idx].astype(np.uint8))

    fig.tight_layout()
    anim = animation.FuncAnimation(fig,
                                    update,
                                    frames=len(frames),
                                    interval=50)
    plt.show()


def main_webcam():
    # Set up VNext predictor
    vnext_cfg = setup_cfg(args)
    vnext_seg_predictor = DefaultPredictor(vnext_cfg)

    # set up models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "e2fgvi":
        size = (432, 240)
    elif args.set_size:
        size = (args.width, args.height)
    else:
        size = None

    net = importlib.import_module('model.' + args.model)
    model = net.InpaintGenerator().to(device)
    data = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(data)
    print(f'Loading model from: {args.ckpt}')
    model.eval()
    
    video_history = []
    mask_imgs = [None] * args.max_frames
    h, w = size[1], size[0]
    neighbor_ids = [*range(args.max_frames - neighbor_stride, args.max_frames)]

    # prepare VNext
    vnext_viz_colormode = ColorMode.IMAGE_BW
    
    #set up webcam https://subscription.packtpub.com/book/application-development/9781785283932/3/ch03lvl1sec28/accessing-the-webcam
    print('Attempting to open webcam')
    cap = cv2.VideoCapture("rtsp://127.0.0.1:8554/webcam", cv2.CAP_FFMPEG)
    
    #check if webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    ret = True
    while ret:
        ret, frame = cap.read()
        frame = cv2.resize(frame, size)
        
        # Populate video history
        if len(video_history) < args.max_frames:
            video_history.append(frame)
            continue

        # Get random reference IDs
        ref_ids = random.sample([*range(1, args.max_frames - neighbor_stride)], num_ref)

        # Prepare a segment of images for this iteration
        cur_frame_id = 0
        segment_ids = neighbor_ids + ref_ids
        segment_frames = [frame] + [video_history[i] for i in segment_ids]
        segment_ids = [cur_frame_id] + segment_ids
        print(f'Using frame IDs {segment_ids} for this process segment')

        #get segmentation/masks using frames
        vnext_masks = []
        mask_img = None
        for idx in range(len(segment_ids)):
            # Only process segmentation for images not processed yet
            if idx != 0 and mask_imgs[segment_ids[idx]] is not None:
                vnext_masks.append(mask_imgs[segment_ids[idx]])
                continue

            predictions = vnext_seg_predictor(segment_frames[idx])
            
            # If we don't have any detections then just use the original frame
            if 'instances' not in predictions or len(predictions['instances']) <= 0:
                vnext_masks.append(segment_frames[idx])
                continue

           # Only use segmentation masks that are not of people (remove occlusions for people)
            instances = predictions['instances'].to(torch.device("cpu"))
            instances = instances[instances.pred_classes != CLASS_PERSON_VALUE]

            # Remove our boxes, classes, and scores from the segmentation instances before
            # we draw our masked images to prevent the labels and boxes from being drawn
            pred_boxes = instances.get('pred_boxes').tensor.numpy() if instances.has('pred_boxes') else None
            pred_scores = instances.get('scores') if instances.has('scores') else None
            pred_classes = instances.pred_classes.tolist() if instances.has("pred_classes") else None

            if instances.has('pred_boxes'):
                instances.remove('pred_boxes')
            if instances.has('scores'):
                instances.remove('scores')
            if instances.has('pred_classes'):
                instances.remove('pred_classes')

            # Get our segmentation masked images
            masked_img = np.zeros_like(segment_frames[idx])
            masked_img = masked_img[:, :, ::-1]
            visualizer = Visualizer(masked_img, None, instance_mode=vnext_viz_colormode)
            vis_frame = visualizer.draw_instance_predictions(predictions=instances)
            vnext_masks.append(cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR))

            if idx == 0:
                mask_img = vnext_masks[-1]
            else:
                mask_imgs[segment_ids[idx]] = vnext_masks[-1]

        masks = preprocess_masks(vnext_masks, size)
        binary_masks = [
            np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks
        ]

        '''# Visualize the masked images
        for idx in range(len(tmp)):
            m = np.array(masks[idx])
            cv2.namedWindow("Mask img: " + str(tmp[idx]), cv2.WINDOW_NORMAL)
            cv2.imshow("Mask img: " + str(tmp[idx]), m)
            if cv2.waitKey(0) == 27:
                break
        '''

        imgs = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in segment_frames]
        imgs = to_tensors()(imgs).unsqueeze(0) * 2 - 1
        masks = to_tensors()(masks).unsqueeze(0)
        imgs, masks = imgs.to(device), masks.to(device)
        res_img = None
        with torch.no_grad():
            masked_imgs = imgs * (1 - masks)
            
            '''# Visualize original images with mask holes
            for idx in range(len(all_ids)):
                m = np.array(masked_imgs.to(torch.device('cpu'))).squeeze()[idx].transpose(1, 2, 0) * 255
                m = cv2.cvtColor(m.astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.namedWindow("Mask img: " + str(all_ids[idx]), cv2.WINDOW_NORMAL)
                cv2.imshow("Mask img: " + str(all_ids[idx]), m)
                if cv2.waitKey(0) == 27:
                    break
            '''

            mod_size_h = 60
            mod_size_w = 108
            h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
            w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
            masked_imgs = torch.cat(
                [masked_imgs, torch.flip(masked_imgs, [3])],
                3)[:, :, :, :h + h_pad, :]
            masked_imgs = torch.cat(
                [masked_imgs, torch.flip(masked_imgs, [4])],
                4)[:, :, :, :, :w + w_pad]
            pred_imgs, _ = model(masked_imgs, len(neighbor_ids))
            
            # Get our last frame prediction, which is our current camera frame
            cur_pred_img = pred_imgs[cur_frame_id, :, :h, :w]
            cur_pred_img = (cur_pred_img + 1) / 2
            cur_pred_img = cur_pred_img.cpu().permute(1, 2, 0).numpy() * 255
            res_img = np.array(cur_pred_img).astype(
                    np.uint8) * binary_masks[cur_frame_id] + frame * (
                        1 - binary_masks[cur_frame_id])

        cv2.imshow('Result', res_img) 
        c = cv2.waitKey(1)
        if c == 27:
            break

        # Remove the oldest frame in our history
        video_history = video_history[1:] + [frame]
        mask_imgs = mask_imgs[1:] + [mask_img]
    
    cap.release()


if __name__ == '__main__':
    if args.webcam == False:
        main_video()
    else:
        main_webcam()
