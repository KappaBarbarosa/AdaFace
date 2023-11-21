from typing import Tuple
import torch
import time
import numpy as np
import cv2
import torch.backends.cudnn as cudnn
from yolo_face.models.experimental import attempt_load
from yolo_face.utils.datasets import LoadStreams, LoadImages,letterbox
from yolo_face.utils.general import check_img_size, non_max_suppression,scale_coords, get_crop
from yolo_face.utils.align_trans import get_reference_facial_points, warp_and_crop_face
def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()
def kpt_tensor_to_list(keypoint_tensor):
    keypoint_list = keypoint_tensor.tolist()
    keypoint_coords = [keypoint_list[i:i+3][:2] for i in range(0, len(keypoint_list), 3)]
    return [coord[:2] for coord in keypoint_coords]
def to_input(pil_rgb_image,RGB=False):
    np_img = np.array(pil_rgb_image)
    if RGB:
        brg_img = ((np_img / 255.) - 0.5) / 0.5
    else :
        brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor(np.expand_dims(brg_img.transpose(2,0,1),axis=0)).float()
    return tensor
class YOLO_FACE:
    def __init__(self, weights='yolov7-tiny.pt',imgsz=640,device=None,crop_size: Tuple[int, int] = (112, 112)):
        # Initialize
        self.device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if device is None else device
        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        if isinstance(imgsz, (list, tuple)):
            imgsz[0] = check_img_size(imgsz[0], s=self.stride)
            imgsz[1] = check_img_size(imgsz[1], s=self.stride)
        else:
            imgsz = check_img_size(imgsz, s=self.stride)  # check img_size
        self.imgsz = imgsz
        self.stride = int(self.model.stride.max())
        self.crop_size = crop_size
        self.refrence = get_reference_facial_points(default_square=crop_size[0] == crop_size[1])

    def detect(self,source):
        self.webcam =  source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        if self.webcam:
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=self.imgsz, stride=self.stride)
        else:
            dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride)
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        crops=[]
        for _, img, im0s, _ in dataset:
            img = torch.from_numpy(img).to(self.device).float() 
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            t1 = time_synchronized()
            pred = self.model(img, augment=False)[0]
            # Apply NMS
            pred = non_max_suppression(pred, 0.25, 0.45)
            t2 = time_synchronized()
            for i, det in enumerate(pred):  # detections per image
                im0 =im0s.copy()
                if self.webcam:  # batch_size >= 1
                    im0 = im0s[i].copy()
                else:
                    im0 = im0s.copy()
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
                    scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=5, step=3)
                    for det_index, (*xyxy, _, _) in enumerate(reversed(det[:, :6])):
                        facial5points = kpt_tensor_to_list(det[det_index, 6:])
                        warped_face = warp_and_crop_face(np.array(im0s), facial5points, self.refrence, crop_size=self.crop_size)
                        crops.append(warped_face)
        #             print(f'Done. ({t2 - t1:.3f}s)')
        # print(f'detect {len(crops)} faces in this source')
        return crops
    def video_detect(self,source,fr_model,view_img,database,ID_list):
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=self.imgsz, stride=self.stride)
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        for _, img, im0s, _ in dataset:
            img = torch.from_numpy(img).to(self.device).float() 
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            pred = self.model(img, augment=False)[0]
            # Apply NMS
            pred = non_max_suppression(pred, 0.25, 0.45)
            for i, det in enumerate(pred):  # detections per image
                im0 = im0s[i].copy()
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
                    scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=5, step=3)
                    for det_index, _ in enumerate(reversed(det[:, :6])):
                        facial5points = kpt_tensor_to_list(det[det_index, 6:])
                        warped_face = warp_and_crop_face(np.array(im0s[i]), facial5points, self.refrence, crop_size=self.crop_size)
                        bgr_tensor_input = to_input(warped_face,True).to(self.device)
                        feature, _ = fr_model(bgr_tensor_input)    
                        similarity_scores = feature @ database.T
                        max_index = torch.argmax(similarity_scores).item()
                        if similarity_scores[0, max_index] >= 0.4:
                            print(f'Detected: {ID_list[int(max_index/3)]}')
                        else:
                            print("No recognized face")           
                if view_img:
                    cv2.imshow('Webcam', im0)
                    cv2.waitKey(1) 
    def frame_detect(self,frame):
        crops=[]
        img = letterbox(frame, self.imgsz, stride=self.stride, auto=False)[0]
        img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1) )
        img = torch.from_numpy(img).to(self.device).float() 
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        pred = self.model(img, augment=False)[0]
        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45)
        for i, det in enumerate(pred):  # detections per image
            im0 =frame.copy()
            if len(det):
                # Rescale boxes from img_size to im0 size
                scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
                scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=5, step=3)
                for det_index, _ in enumerate(reversed(det[:, :6])):
                    facial5points = kpt_tensor_to_list(det[det_index, 6:])
                    warped_face = warp_and_crop_face(np.array(frame), facial5points, self.refrence, crop_size=self.crop_size)
                    crops.append(warped_face)
        return crops

if __name__ == '__main__':
    with torch.no_grad():
        yoloface = YOLO_FACE('yolo_face/yolov7-tiny.pt',640)
        crops = yoloface.detect('yolo_face/data/images/test07.jpg')
        print(len(crops))
