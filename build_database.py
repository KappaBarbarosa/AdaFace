import net
import torch
import os
from yolo_face.align import YOLO_FACE
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
adaface_models = {
    'ir_50':"retrained/epoch=14-step=166804.ckpt",
    'ir_101':"pretrained/adaface_ir101_ms1mv2.ckpt",
    'epoch14':"pretrained/epoch=14-step=166804.ckpt"
}

def load_pretrained_model(architecture='ir_50',path=''):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    pt = path if path is  not None else adaface_models[architecture]
    statedict = torch.load(pt)['state_dict']
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
def save_database_heatmap(data,savepath):
    plt.figure(figsize=(10, 10))
    plt.imshow(data, cmap='viridis', interpolation='nearest')
    for i in range(len(data)):
        for j in range(len(data[0])):
            plt.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color='w')
    plt.colorbar()
    plt.savefig(savepath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fr_weights', nargs='+', type=str, default='pretrained/epoch=14-step=166804.ckpt', help='face recongnition model.ckpt path(s)')
    parser.add_argument('--fd_weights', nargs='+', type=str, default='pretrained/yolov7-tiny.pt', help='face detection model.pt path(s)')
    parser.add_argument('--arch', type=str,default='ir_50',help='Adaface architecture')
    parser.add_argument('--database_path', type=str,default='face_database',help='face database path')
    parser.add_argument('--database_savedir', type=str,default='face_database',help='face database ckpt save path')
    args = parser.parse_args()
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pretrained_model(args.arch,path=args.fr_weights).to(device)
    print('face recongnition model loaded')
    ID_list_dir = os.path.join(args.database_path,'ID') 
    features = []
    ID_list = sorted(os.listdir(ID_list_dir))
    start = time.time()
    yoloface = YOLO_FACE(args.fd_weights,device=device)
    print('face detection model loaded')
    start = time.time()
    for ID in ID_list:
        path = os.path.join(ID_list_dir, ID)
        aligned_bgr_imgs=yoloface.detect(path)
        for i,img in enumerate(aligned_bgr_imgs) :
            tensor_img = to_input(img,True).to(device)
            with torch.no_grad():
                feature, _ = model(tensor_img)
            features.append(feature)

    print(f'building time: {time.time()-start}')
    features = torch.cat(features)
    if not os.path.exists(args.database_savedir):
        os.makedirs(args.database_savedir)
        print(f"Directory '{args.database_savedir}' created.")
    pt_spath = os.path.join(args.database_savedir,'database.pt')
    heatmap_spath = os.path.join(args.database_savedir,'heatmap.jpg')
    IDlist_spath = os.path.join(args.database_savedir,'ID_list.txt')
    torch.save(features,pt_spath)
    similarity_scores = features @ features.T
    save_database_heatmap(similarity_scores.cpu().detach().numpy(),savepath=heatmap_spath)
    with open(IDlist_spath, 'w') as file:
        for item in ID_list:
            file.write("%s\n" % item)
    print(f'build success, save database reslut in {args.database_savedir}')

    

