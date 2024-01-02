import net
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from yolo_face.align import YOLO_FACE
import warnings
import sys
# ignore all the SyntaxWarning
warnings.filterwarnings("ignore", category=SyntaxWarning)

adaface_models = {
    'ir_50':"pretrained/epoch=14-step=166804.ckpt",
    # 'ir_50':"pretrained/adaface_ir50_ms1mv2.ckpt",
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
    model = load_pretrained_model('ir_50').to(device)
    database_dir = './face_database/database.pt'
    if(os.path.exists(database_dir)):
        database = torch.load(database_dir).to(device)
        ID_list = sorted(os.listdir('face_database/ID'))
        print("ID_list's len: ", len(ID_list))
    else:
        print("ERROR: Face database not found. Please create a face database first.")
        sys.exit(1)
    # test_image_path = 'face_alignment/test_images'
    # cap = cv2.VideoCapture(0) # load from webcam
    features = []
    face=[]
    isyolo = True
    yoloface = YOLO_FACE('yolo_face/yolov7-tiny.pt',device=device) if isyolo else None
    yoloface.video_detect(0,fr_model= model,view_img= True,database=database,ID_list=ID_list)