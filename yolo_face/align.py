from typing import Tuple
import torch
import time
import numpy as np
import cv2
import torch.backends.cudnn as cudnn
from yolo_face.models.experimental import attempt_load
from yolo_face.utils.datasets import LoadStreams, LoadImages,letterbox
from yolo_face.utils.general import check_img_size, non_max_suppression,scale_coords, get_crop, save_one_box
from yolo_face.utils.align_trans import get_reference_facial_points, warp_and_crop_face
from yolo_face.utils.plots import colors, plot_one_box
# from ..inference_video import student_arrive, create_class
# firebase functions
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import datetime
import pytz
import asyncio


def create_class(class_date, db):
    student_id_list_ref = db.collection("students").document("student_id_list")
    class_doc_ref = db.collection("classes").document(str(class_date))
    class_doc_ref.set(student_id_list_ref.get().to_dict())
    student_id_list = student_id_list_ref.get().to_dict()
    for student_id in student_id_list.keys():
        student_document_ref = db.collection("students").document("student_id_list").collection("ML").document(student_id)
        student_update_data = {
            class_date : None
        }
        if student_document_ref.get().exists == False:
            student_document_ref.set(student_update_data)

def student_arrive(student_id, db):
    cur_date = datetime.now(pytz.timezone('Asia/Taipei')).strftime("%m%d")
    cur_time = datetime.now(pytz.timezone('Asia/Taipei')).strftime("%H:%M:%S")
    class_ref = db.collection("classes").document(str(cur_date))
    snapshot = class_ref.get()
    snapshot_dict = snapshot.to_dict()
    class_update_data = {
        student_id: cur_time
    }
    if class_ref.get().exists == False:
        create_class(cur_date, db)
    elif type(snapshot_dict[student_id]) == str:
        print(f"{student_id} had arrived at {snapshot_dict[student_id]}")
        return
    class_ref.update(class_update_data)
    ##########
    student_document_ref = db.collection("students").document("student_id_list").collection("ML").document(student_id)
    student_update_data = {
        cur_date : cur_time
    }
    if student_document_ref.get().exists == False:
        student_document_ref.set(student_update_data)
    else:
        student_document_ref.update(student_update_data)

    print(f"update successed. {student_id} arrives at {cur_date} {cur_time}")



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
            pred = self.model(img, augment=False)[0]
            # Apply NMS
            pred = non_max_suppression(pred, 0.25, 0.45)
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
        return crops
    def show_detect(self,source):
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
        tests=[]
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
                im1 =im0s.copy()
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
                        tests.append(warp_and_crop_face(np.array(im1), facial5points, self.refrence, crop_size= (224, 224)))
        #             print(f'Done. ({t2 - t1:.3f}s)')
        # print(f'detect {len(crops)} faces in this source')
        return crops,tests
    def video_detect(self,source,fr_model,view_img,database,ID_list):
        local_ID = [0] * len(ID_list)
        cam_start_time = time.time()
        cred = credentials.Certificate("./serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
        db = firestore.client()
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
                    for det_index, (*xyxy, conf, cls)in enumerate(reversed(det[:, :6])):
                        kpts = det[det_index, 6:]
                        
                        facial5points = kpt_tensor_to_list(det[det_index, 6:])
                        warped_face = warp_and_crop_face(np.array(im0), facial5points, self.refrence, crop_size=self.crop_size)
                        bgr_tensor_input = to_input(warped_face,True).to(self.device)
                        feature, _ = fr_model(bgr_tensor_input)    
                        similarity_scores = feature @ database.T
                        max_index = torch.argmax(similarity_scores).item()
                        if max_index == 36:
                            max_index -= 1
                        if similarity_scores[0, max_index] >= 0.4:
                            print("max index: ", max_index)
                            print(f'Detected: {ID_list[int(max_index/3)]}')
                            # print("xyxy= ", xyxy)
                            cur_time = time.time()
                            if cur_time - cam_start_time >= 10:
                                if local_ID[int(max_index/3)] == 0:
                                    print("in")
                                    student_arrive(str(ID_list[int(max_index/3)]), db)
                                    local_ID[int(max_index/3)] = 1
                                cam_start_time = cur_time
                            # save_one_box(xyxy, im0, file=ID_list[int(max_index/3)])
                            plot_one_box(xyxy, im0, label=ID_list[int(max_index/3)], color=colors(int(max_index/3), True), kpts=kpts, steps=3, orig_shape=im0.shape[:2])
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
    def loop_detect_with_frmodel(self):
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
