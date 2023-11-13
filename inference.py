import net
import torch
import os
from face_alignment import align
import numpy as np
import time
import matplotlib.pyplot as plt
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

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
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
    test_image_path = 'face_alignment/test_images'
    features = []
    test_image = sorted(os.listdir(test_image_path))
    for fname in test_image:
        path = os.path.join(test_image_path, fname)
        with torch.no_grad():
            print(f'detecting {path}s')
            aligned_rgb_img = align.get_aligned_face(path, isarray=True)
            if(aligned_rgb_img is not None):
                bgr_tensor_input = to_input(aligned_rgb_img).to(device)
                feature, _ = model(bgr_tensor_input)
                features.append(feature)
    database_dir = 'face_database/database.pt'
    if(os.path.exists(database_dir)):
        database = torch.load(database_dir).to(device)
        ID_list = sorted(os.listdir('face_database/ID'))
        start_time = time.time()
        with torch.no_grad():
            similarity_scores = torch.cat(features) @ database.T
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"computing time: {execution_time} seconds")
        score = similarity_scores.cpu().detach().numpy()
        save_database_heatmap(score)
        max_index = np.argmax(score, axis=1)
        for i,id in enumerate(max_index):
            print(test_image[i])
            if(score[i][id] >= 0.4):
                print(f'detetct {ID_list[int(id/3)]} with similarity score {score[i][id]}')
            else:
                print("no ensured answer")
    else:
        with torch.no_grad():
            similarity_scores = torch.cat(features) @ torch.cat(features).T
        print(similarity_scores)
    

