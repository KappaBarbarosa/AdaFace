from pathlib import Path
import argparse
import mxnet as mx
from tqdm import tqdm
from PIL import Image
import bcolz
import pickle
import cv2
import numpy as np
from torchvision import transforms as trans
from MaskTheFace.utils.aux_functions import *
import AdaFace.evaluate_utils as evaluate_utils
import os
import numbers
import dlib
import random
import json
import ast
types=['surgical','cloth']
colors = ["#000000","#FFFFFF","#99FFFF","#99CCFF","#808080"]
def save_rec_to_img_dir(rec_path,args, swap_color_channel=False, save_as_png=False):

    save_path = rec_path/'imgs'
    if not save_path.exists():
        save_path.mkdir()
    imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path/'train.idx'), str(rec_path/'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    label_ct={}
    save_txt = rec_path/'dic.txt'
    if not save_txt.exists():
        for idx in tqdm(range(1,max_idx)):
            img_info = imgrec.read_idx(idx)
            header, img = mx.recordio.unpack_img(img_info)
            if not isinstance(header.label, numbers.Number):
                label = int(header.label[0])
            else:
                label = int(header.label)
            if label not in label_ct:
                label_ct[label] = 1
            else:
                label_ct[label]+=1
        json_str = json.dumps(label_ct, indent=4)
        with open(save_txt, 'w') as file:
            file.write(json_str)
    else:
        with open(save_txt, 'r') as file:
            content = file.read()
            label_ct = ast.literal_eval(content) 
            print(label_ct['20'])
    limit_ct ={}  
    with open('/content/drive/MyDrive/data/faces_glintasia/correct_dir.json', 'r') as json_file:
       correct_dict = json.load(json_file)
    for idx in tqdm(range(1,max_idx)):
        if idx < 818252:
            continue
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        if not isinstance(header.label, numbers.Number):
            label = str(int(header.label[0]))
        else:
            label = str(int(header.label))
        if (label in correct_dict):
            if(correct_dict[label]) == True:
             continue
        if label not in limit_ct:
            tomask = True
            limit_ct[label] = 1
        else:
            tomask = limit_ct[label] < int(args.masked_ratio*label_ct[label])
            limit_ct[label]+=1
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if tomask:
            rid = random.randint(0, 1)
            cid = random.randint(0, 4)
            args.color  = colors[cid]
            args.mask_type = types[rid]
            masked_image, mask, mask_binary_array, original_image = mask_image(img, args)
            if len(masked_image) >= 1: 
                img = masked_image[0]
            else:
              img = original_image 
              limit_ct[label]-=1
        if swap_color_channel:
            # this option saves the image in the right color.
            # but the training code uses PIL (RGB)
            # and validation code uses Cv2 (BGR)
            # so we want to turn this off to deliberately swap the color channel order.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
        img = Image.fromarray(img)
        label_path = save_path/str(label)
        if not label_path.exists():
            label_path.mkdir()

        if save_as_png:
            img_save_path = label_path/'{}.png'.format(idx)
            img.save(img_save_path)
        else:
            img_save_path = label_path/'{}.jpg'.format(idx)
            img.save(img_save_path, quality=95)

def load_bin(path, rootdir, image_size=[112,112]):

    test_transform = trans.Compose([
                trans.ToTensor(),
                trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    if not rootdir.exists():
        rootdir.mkdir()
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    # data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')
    for i in tqdm(range(len(bins))):
        if((i<20) | (i>(len(bins)-30))):
            _bin = bins[i]
            img = mx.image.imdecode(_bin).asnumpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = Image.fromarray(img.astype(np.uint8))
            # data[i, ...] = test_transform(img)
            print(f'{rootdir}/{i}.jpg')
            img.save(f'{rootdir}/{i}.jpg')
        i += 1
    print(issame_list[:20])
    print(issame_list[-20:])
    # print(data.shape)
    # np.save(str(rootdir)+'_list', np.array(issame_list))
    return  issame_list
def load_bin_withmask(path, rootdir,args, image_size=[112,112]):
    test_transform = trans.Compose([
                trans.ToTensor(),
                trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    if not rootdir.exists():
        rootdir.mkdir()
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')
    mask_idx = len(bins)* args.masked_ratio
    actual_mask=0
    print(f'take {mask_idx} / {len(bins)} as masked img in {rootdir}')
    for i in tqdm(range(len(bins))):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if i < mask_idx :
            rid = random.randint(0, 1)
            cid = random.randint(0, 4)
            args.mask_type = 'surgical' if rid == 0 else 'cloth'
            args.color  = colors[cid]
            masked_image, mask, mask_binary_array, original_image = mask_image(img, args)
            if len(masked_image) == 1: 
                img = masked_image[0]
                actual_mask+=1
            else:
              img = original_image 
              mask_idx+=1 
        img = Image.fromarray(img.astype(np.uint8))
        data[i, ...] = test_transform(img)
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
            img.save(f'{rootdir}/{i}.jpg')
    print(data.shape)
    print(f'{actual_mask} faces are masked')
    np.save(str(rootdir)+'_list', np.array(issame_list))
    return data, issame_list
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-r", "--rec_path", help="mxnet record file path", default='./faces_emore', type=str)
    parser.add_argument("--make_image_files", action='store_true')
    parser.add_argument("--make_validation_memfiles", action='store_true')
    parser.add_argument("--swap_color_channel", action='store_true')
    parser.add_argument(
    "--mask_type",
    type=str,
    default="surgical",
    choices=["surgical", "N95", "KN95", "cloth", "gas", "inpaint", "random", "all"],
    help="Type of the mask to be applied. Available options: all, surgical_blue, surgical_green, N95, cloth",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="",
        help="Type of the pattern. Available options in masks/textures",
    )
    parser.add_argument(
        "--pattern_weight",
        type=float,
        default=0.5,
        help="Weight of the pattern. Must be between 0 and 1",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="#0473e2",
        help="Hex color value that need to be overlayed to the mask",
    )
    parser.add_argument(
        "--color_weight",
        type=float,
        default=0.8,
        help="Weight of the color intensity. Must be between 0 and 1",
    )
    parser.add_argument(
        "--masked_ratio",
        type=float,
        default=0.5,
        help="Weight of the color intensity. Must be between 0 and 1",
    )
    parser.add_argument(
        "--code",
        type=str,
        # default="cloth-masks/textures/check/check_4.jpg, cloth-#e54294, cloth-#ff0000, cloth, cloth-masks/textures/others/heart_1.png, cloth-masks/textures/fruits/pineapple.png, N95, surgical_blue, surgical_green",
        default="",
        help="Generate specific formats",
    )
    parser.add_argument(
        "--verbose", dest="verbose", action="store_true", help="Turn verbosity on"
    )
    args = parser.parse_args()
    args.detector = dlib.get_frontal_face_detector()
    path_to_dlib_model = "MaskTheFace/dlib_models/shape_predictor_68_face_landmarks.dat"
    args.predictor = dlib.shape_predictor(path_to_dlib_model)
    rec_path = Path(args.rec_path)
    if args.make_image_files:
        # unfolds train.rec to image folders
        save_rec_to_img_dir(rec_path,args=args, swap_color_channel=args.swap_color_channel)

    if args.make_validation_memfiles:
        # for saving memory usage during training
        bin_files = list(filter(lambda x: os.path.splitext(x)[1] in ['.bin'], os.listdir(args.rec_path)))
        bin_files = [i.split('.')[0] for i in bin_files]
        for i in range(len(bin_files)):
            load_bin_withmask(rec_path/(bin_files[i]+'.bin'), rec_path/bin_files[i],args)
        concat_mem_file_name = os.path.join(args.rec_path, 'concat_validation_memfile')
        print("preparing_data...")
        # call this once to convert val_data to memfile for saving memory
        if not os.path.isdir(os.path.join(args.rec_path, 'agedb_30', 'memfile')):
            print('making validation data memfile')
        evaluate_utils.get_val_data(args.rec_path)

        if not os.path.isfile(concat_mem_file_name):
            print("creating a concat memfile...")
            # create a concat memfile
            concat = []
            for key in ['agedb_30', 'cfp_fp', 'lfw', 'cplfw', 'calfw']:
                np_array, issame = evaluate_utils.get_val_pair(path=args.rec_path,
                                                                name=key,
                                                                use_memfile=True)
                concat.append(np_array)
            concat = np.concatenate(concat)
            evaluate_utils.make_memmap(concat_mem_file_name, concat)

