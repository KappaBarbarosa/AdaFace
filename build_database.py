import net
import torch
import os
from face_alignment import align
from yolov7_face.align import YOLO_FACE
import numpy as np
import matplotlib.pyplot as plt
import time
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

def to_input(pil_rgb_image,RGB=True):
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
    plt.savefig('face_database/heatmap.png')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pretrained_model('ir_50').to(device)
    print('face recongnition model loaded')
    ID_list_dir = 'face_database/ID'
    features = []
    ID_list = sorted(os.listdir(ID_list_dir))
    isyolo = True
    start = time.time()
    if isyolo:
        yoloface = YOLO_FACE('yolov7_face/yolov7-tiny.pt',device=device)
        print('face detection model loaded')
        start = time.time()
        for ID in ID_list:
            path = os.path.join(ID_list_dir, ID)
            aligned_bgr_imgs =yoloface.detect(path)
            for i,img in enumerate(aligned_bgr_imgs) :
                tensor_img = to_input(img,True).to(device)
                with torch.no_grad():
                    feature, _ = model(tensor_img)
                features.append(feature)
    else:
        for ID in ID_list:
            path = os.path.join(ID_list_dir, ID)
            for img_path in sorted(os.listdir(path)):
                with torch.no_grad():
                    aligned_rgb_img = align.get_aligned_face(os.path.join(path, img_path))
                    if aligned_rgb_img is None: continue
                    bgr_tensor_input = to_input(aligned_rgb_img,False).to(device)
                    feature, _ = model(bgr_tensor_input)
                features.append(feature)
    print(f'building time: {time.time()-start}')
    database_dir = 'face_database/database_ir50.pt'
    features = torch.cat(features)
    # torch.save(features,database_dir)
    similarity_scores = features @ features.T
    save_database_heatmap(similarity_scores.cpu().detach().numpy())
    print('build success')

    

