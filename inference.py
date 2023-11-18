import net
import torch
import os
from face_alignment import align
import numpy as np
import time
import matplotlib.pyplot as plt
from yolo_face.align import YOLO_FACE
import cv2
import warnings
import sys

# ignore all the SyntaxWarning
warnings.filterwarnings("ignore", category=SyntaxWarning)

adaface_models = {
    'ir_50':"pretrained/adaface_ir50_ms1mv2.ckpt",
    'ir_101':"pretrained/adaface_ir101_ms1mv2.ckpt",
}

def load_pretrained_model(architecture='ir_50'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image,RGB=False):
    np_img = np.array(pil_rgb_image)
    if RGB:
        brg_img = ((np_img / 255.) - 0.5) / 0.5
    else :
        brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor(np.expand_dims(brg_img.transpose(2,0,1),axis=0)).float()
    return tensor

def save_database_heatmap(data):
    plt.figure(figsize=(10, 10))
    plt.imshow(data, cmap='viridis', interpolation='nearest')
    for i in range(len(data)):
        for j in range(len(data[0])):
            plt.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color='w')
    plt.colorbar()
    plt.savefig('iference_heatmap_101.png')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pretrained_model('ir_101').to(device)
    database_dir = 'face_database/database.pt'
    if(os.path.exists(database_dir)):
        database = torch.load(database_dir).to(device)
        ID_list = sorted(os.listdir('face_database/ID'))
    else:
        print("ERROR: Face database not found. Please create a face database first.")
        sys.exit(1)
    features = []
    face=[]
    isyolo = True
    cap = cv2.VideoCapture(0) # load from webcam
    yoloface = YOLO_FACE('yolo_face/yolov7-tiny.pt',device=device) if isyolo else None
    ct=0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        with torch.no_grad():
            aligned_rgb_img = align.get_aligned_face(frame, isarray=True)
            if aligned_rgb_img is not None:
                bgr_tensor_input = to_input(aligned_rgb_img, False).to(device)
                feature, _ = model(bgr_tensor_input)

                # calculate score and recognize
                with torch.no_grad():
                    similarity_scores = feature @ database.T
                max_index = torch.argmax(similarity_scores).item()
                #print(f'Similarity Score: {similarity_scores[0, max_index]:.2f}')
                if similarity_scores[0, max_index] >= 0.4:
                    print(f'Detected: {ID_list[max_index]}')
                else:
                    print("No recognized face")
        cv2.imshow('Webcam', frame)  
        # terminate
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()