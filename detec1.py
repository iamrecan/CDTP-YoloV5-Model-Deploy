import argparse
import os
import shutil
import time
from pathlib import Path

import json

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, strip_optimizer, set_logging)
from utils.torch_utils import select_device


def detect(save_img=False):
    # Get setting parameter data
    out, source, weights, view_img, save_txt, imgsz = \
        opt.save_dir, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):  # output dir
        shutil.rmtree(out)  # delete dir
    os.makedirs(out)  # make new dir
    #  Use float16 if the device is GPU
    half = device.type != 'cpu'  # half precision only supported on CUDA

    #  Load Model Make sure that the user setting input picture resolution can be removed 32 (if it is not possible to adjust to removal and return)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    #  Second-Stage Classifier Sets the second classification, not used by default
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    #  Set DataLoader Set different data load by different input sources
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    #  Get category name
    names = model.module.names if hasattr(model, 'module') else model.names
    #  Set the color of the picture frame
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    #  Take a forward reasoning, whether the test program is normal
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    #  Output JSON file
    save_json = True
    content_json = []

    #  Path Picture / Video Path
    #  Image after IMG for Resize + Pad
    #  IMG0 original size picture
    #  CAP When reading a picture, a video source is read when reading a video.
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        #  The picture is also set to float16
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        #  If there is no batch_size, add an axis in front.
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        """
                 Before the forward spread, the Shape is (1, Num_boxes, 5 + Num_Class)
                 H, W is the length and width of the incoming network picture, pay attention to the rectangular reasoning when DataSet is detected, so h is not equal to W
        num_boxes = h/32 * w/32 + h/16 * w/16 + h/8 * w/8
                 PRED [..., 0: 4] for prediction box coordinates
                 Prediction box coordinates for XYWH (central point + width) format
                 PRED [..., 4] confessed to Objectness
                 PRED [..., 5: -1] is classified results
        """
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        """
                 PRED: Output of forward propagation
                 Conf_thres: confidence threshold
                 IOU_THRES: IOU threshold
                 Classes: Do you only keep a specific category?
                 Agnostic: Does NMS also remove box between different categories
                 After the NMS, the prediction box format: XYWH -> XYXY (left upper corner right corner)
                 PRED is a list list (Torch.Tensor], the length is BATCH_SIZE
                 Every Torch.Tensor's Shape is (Num_boxes, 6), the content is Box + Conf + CLS
        """
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        #  Add secondary classification, not used by default
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        #  Treatment for each picture
        for i, det in enumerate(pred):  # detections per image
            #  If the input source is Webcam, Batch_size is not 1, remove a picture in the DataSet
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            #  Set the path to save image / video
            save_path = str(Path(out) / Path(p).name)
            #  Set the path to the save box coordinate TXT file
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            #  Set print information (picture long width)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                #  Adjust the coordinates of the prediction box: Coordinate of the picture of Resize + Pad -> Coordinate based on the original size picture
                #  At this point, the coordinate format is XYXY.
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                #  Print the number of categories
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        #  Turn XYXY (left upper corner + upper right corner) format to XYWH (central point + wide length) format, and in addition to W, H is normalized, convert to list and save
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, conf, *xywh) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            #  Picture in the original picture
                            f.write(('%g ' * len(line) + '\n') % line)

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                    #  Output JSON file
                    if save_json:
                        #  Use under Windows
                        file_name = save_path.split('\\')
                        #  Used under Linux
                        # file_name = save_path.split('/')
                        content_dic = {
                            "name": file_name[len(file_name)-1],
                            "category": (names[int(cls)]),
                            "bbox": torch.tensor(xyxy).view(1, 4).view(-1).tolist(),
                            "score": conf.tolist()
                        }
                        content_json.append(content_dic)

            # Print time (inference + NMS)
            #  Print forward propagation time
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            #  If set, show pictures / videos
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            #  Set saving image / video
            # if save_img:
            #     if dataset.mode == 'images':
            #         cv2.imwrite(save_path, im0)
            #     else:
            #         if vid_path != save_path:  # new video
            #             vid_path = save_path
            #             if isinstance(vid_writer, cv2.VideoWriter):
            #                 vid_writer.release()  # release previous video writer
            #
            #             fourcc = 'mp4v'  # output video codec
            #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            #         vid_writer.write(im0)

    if save_txt or save_img or save_json:
        print('Results saved to %s' % Path(out))
        #  Write JSON data into a file
        with open(os.path.join(Path(out), 'result.json'), 'w') as f:
            json.dump(content_json, f)
    #  Print total time
    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    """
         Weights: Weight of training
         Source: Test data, can be a picture / video path, or '0' (computer comes with camera), or RTSP and other video streams
         Output: Picture / Video Save Path after Network Forecast
         IMG-SIZE: Network Enter picture size
         Conf-thres: confidence threshold
         IOU-THRES: IOU threshold for NMS
         Device: Setting up the device
         View-img: Whether to show the picture / video after forecast, default false
         Save-txt: Whether to save the predicted frame coordinate in the form of a TXT file, default false
         Classes: Settings only keeps a part of the category, such as 0 or 0 2 3
         Agnostic-NMS: Does NMS also remove box between different categories, default false
         Augment: Multi-scale, flip and other operations (TTA) reasoning when reasoning
         UPDATE: If you are True, all models are strip_optimizer operations, remove information such as optimizer in the PT file, and default is false
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='../tile/testA_imgs', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=1600, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-dir', type=str, default='detect_img/output', help='directory to save results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')

    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                #  Remove information such as optimizer in the PT file
                strip_optimizer(opt.weights)
        else:
            detect()