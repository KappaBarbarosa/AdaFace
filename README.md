# Masked AdaFace


# Introduction
The github repository for the final project of 2023 NTHU Machine Learning about Upgraded AdaFace.

# Installation

```
conda create --name adaface pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
conda activate adaface
conda install scikit-image matplotlib pandas scikit-learn 
pip install -r requirements.txt
```



# Training 
## Preparing Dataset
### For Traing and High Quality Validation datasets
1. Download the dataset from [insightface link](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)
2. Unzip it to a desired location, `DATASET_ROOT`  _ex)_ `/data/`.
3. The result folder we will call `DATASET_NAME`, ex) `faces_webface_112x112`.
### For Mixed Quality Validation datasets
1. Download the dataset from [ijb](https://github.com/deepinsight/insightface/tree/master/recognition/_evaluation_/ijb)
2. Move the Updated Meta data into corresponding folder
## Masking Datasets
### API Installation
```
git clone https://github.com/aqeelanwar/MaskTheFace.git
pip install -r MaskTheFace/requirements.txt
```
### For Traing and High Quality Validation datasets
`python convert.py --rec_path <DATASET_ROOT>/<DATASET_NAME> --make_image_files --make_validation_memfiles`
### For Mixed Quality Validation datasets
`python convert_IJB.py --rec_path <DATASET_ROOT>/<DATASET_NAME>`
| Argument | Explanation                                                                                            |
|------|----------------------------------------------------------------------------------------------------------|
| dataset_path | Path to the image file or a folder containing images to be masked.     | 
| mask_type  | Select the mask to be applied. Available options are 'N95', 'surgical_blue', 'surgical_green', 'cloth', 'empty' and 'inpaint'. The details of these mask types can be seen in the image above. More masks will be added     | 
| pattern  | Selects the pattern to be applied to the mask-type selected above. The textures are available in the masks/textures folder. User can add more textures.|
| pattern_weight  | Selects the intensity of the pattern to be applied on the mask. The value should be between 0 (no texture strength) to 1 (maximum texture strength)|
| color  | Selects the color to be applied to the mask-type selected above. The colors are provided as hex values.|
| color_weight  | Selects the intensity of the color to be applied on the mask. The value should be between 0 (no color strength) to 1 (maximum color strength)|
| masked_ratio  | Select the proportion of people being masked in the dataset|
| code  | Can be used to create specific mask formats at random. More can be found in the section below.|
| verbose  | If set to True, will be used to display useful messages during masking|
## Training
- Sample run scripts are provided in `script`
- EX) Run `bash script/run_ir50_ms1mv2.sh` after changing the `--data_root` and `--train_data_path` to fit your needs. 
* [IMPORTANT] Once the training script has started, check if your image color channel is correct by looking at the sample stored in `<RUN_DIR>/training_samples`. 

# Our Pretrained Models

Note that our pretrained model takes the input in BGR color channel. 
This is different from the InsightFace released model which uses RGB color channel. 

| Arch | Dataset    | Link                                                                                         |
|------|------------|----------------------------------------------------------------------------------------------|
| R50  | Masked CASIA-WebFace     | [gdrive](https://drive.google.com/file/d/1KhCoyNUITaTh3Ns8r76FDR3H3vyuUjOk/view?usp=sharing) |
| R50  | Maked MS1MV2     | [gdrive](https://drive.google.com/file/d/1wG1cZO8o20b9jJB81J1ax14kWXDSRsLZ/view?usp=sharing) |

# Our masked data

Note that our pretrained model takes the input in BGR color channel. 
This is different from the InsightFace released model which uses RGB color channel. 

| Dataset    | Link                                                                                         |
|-----------------------------|----------------------------------------------------------------------------------------------------|
|CASIA-Webface | [gdrive](https://drive.google.com/drive/folders/1YKu_DJZ_6rSHsZZEMQ6DOdkrUFg-KeET?usp=drive_link)|
| High Quality Valadaion data | [gdrive](https://drive.google.com/drive/folders/1iBhTtkTfr3We_jbXsTvnZTEmbehuTJjZ?usp=drive_link) |
| Asian-Celeb     | [gdrive](https://drive.google.com/file/d/1Ih9zxLuqUQOe7d9RcfWd_zU-9cpb6ecF/view?usp=sharing) |
| IJBC-C          |  [gdrive](https://drive.google.com/file/d/11oIa2LVdI-Y6YDHmRYBSb-L9FcRCla6X/view?usp=sharing) |
| IJBC-B          |  [gdrive](https://drive.google.com/file/d/1D2ur3Yh0Kl11g6aXPPIAdABQfrBTyyfV/view?usp=sharing) |
|MS1MV2           | Due to work  statation issue, it will be uploaded later.|
                        

# Inference
## Download Neccessary Model Weights
- [yoloface-v7 github](https://github.com/derronqi/yolov7-face)
- [AdaFace github](https://github.com/mk-minchul/AdaFace)

## Build Recognition Face Database
- Save one photo of faces of specific person under **./face_database/ID/[ID]** folder.
- Restrictions of photos:
    1. Contains only one face.
    2. Faced forward.
    3. Without mask.
    4. File name contains only English, name can be arbitrary.
- Then run below instruction.
```
python build_database.py --database_savedir ./face_database
```


## Inference by Video
After building database, if webcam is available:
```
python inference_video.py -- source 0 \\ your webcam number
```
If yoy want to take raw images as input:
```
python inference_frame.py -- source test_images \\ image directory
```
below is the code snippet, which can be considered the core of the entire program.
```python
for det_index, (*xyxy, conf, cls)in enumerate(reversed(det[:, :6])):
    kpts = det[det_index, 6:]
    facial5points = kpt_tensor_to_list(det[len(det) - det_index - 1, 6:])
    warped_face = warp_and_crop_face(np.array(im0), facial5points, self.refrence, crop_size=self.crop_size)
    bgr_tensor_input = to_input(warped_face, True).to(self.device)
    feature, _ = fr_model(bgr_tensor_input)
    similarity_scores = feature @ database.T
```
- Note that **det** is the variable storing information about detected objects (faces) during the video detection process by the FP32 model.
- Facial Key Points Extraction:
It retrieves the facial key points (kpts) associated with the detected face from the prediction results.
- Face Warping and Cropping:
Using the facial key points, the code employs the warp_and_crop_face function to warp and crop the face region from the original image (im0).
- Calculating similarity:
Utilizes matrix multiplication (@ operator) to compute the dot product of the feature vector with each feature vector in the database.
## Example
- ![image](https://github.com/KappaBarbarosa/AdaFace/blob/master/demo_video1.gif)
- ![image](https://github.com/KappaBarbarosa/AdaFace/blob/master/demo_video2.gif)
## Connect to Firebase
- Follow the instructions of [firebase official website](https://firebase.google.com/docs/firestore/quickstart?hl=zh-cn).
- To get *serviceAccountKey.json*: 
    - click "產生新的私密金鑰" 
    ![image]([https://hackmd.io/_uploads/BykMVnIPa.png](https://github.com/KappaBarbarosa/AdaFace/blob/master/image.png))
    - store the generated file as *serviceAccountKey.json* under this directory.
- After above operations, run python code below to initialize your database.
```
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate("./serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()
# put your ID_list in it.
users = {
    "id1":True,
    "id2":True,
    "id3":True,
    "...":True
}

student_id_list_ref = db.collection("students").document("student_id_list")
student_id_list_ref.set(users)
```

- The structure of database is adjustable, with also adjust the *create_class()* and *student_arrive()* function in *align.py*.
- Then design an website using [firebase hosting](https://firebase.google.com/docs/hosting/quickstart?hl=zh-cn).
- For the website we deigned, refer to [Group5 Attendence Record](https://mltest-26ce3.web.app/).

## High Quality Image Validation Sets (LFW, CFPFP, CPLFW, CALFW, AGEDB)
For evaluation on 5 HQ image validation sets with pretrained models,
refer to 
```
bash validation_hq/eval_5valsets.sh
```

| Arch | Dataset        | Method   | LFW    | CFPFP  | CPLFW   | CALFW   | AGEDB  | AVG       |
|------|----------------|----------|--------|--------|---------|---------|--------|-----------|
| R50  | CASIA-WebFace	 | AdaFace  | 0.9942 | 0.9641 | 0.8997  | 0.9323  | 0.9438 | 0.9468    |
| R50  | CASIA-WebFace (masked) | AdaFace  | 0.9906 | 0.9315 | 0.8765  | 0.9306  | 0.9195 | 0.9297    |
| R50  | MS1MV2         | AdaFace  | 0.9982 | 0.9786 | 0.9283  | 0.9607  | 0.9785 | 0.9688    |
| R50  | MS1MV2 (masked)     | AdaFace  | 0.9980 | 0.9497 | 0.9181  | 0.9597  | 0.9725 | 0.9596    |

## High Quality Masked Image Validation Sets (LFW, CFPFP, CPLFW, CALFW, AGEDB)
For evaluation on 5 HQ masked image validation sets with pretrained models,
refer to 
```
bash validation_hq/eval_5valsets.sh
```

| Arch | Dataset        | Method   | LFW    | CFPFP  | CPLFW   | CALFW   | AGEDB  | AVG       |
|------|----------------|----------|--------|--------|---------|---------|--------|-----------|
| R50  | CASIA-WebFace	 | AdaFace  | 0.9758 | 0.8959 | 0.8495  | 0.9006  | 0.8775 | 0.8999    |
| R50  | CASIA-WebFace (masked) | AdaFace  | 0.9893 | 0.9144 | 0.8658  | 0.9250  | 0.9077 | 0.9204    |
| R50  | MS1MV2         | AdaFace  | 0.9925 | 0.9256 | 0.9038  | 0.9478  | 0.9477 | 0.9435    |
| R50  | MS1MV2 (masked)     | AdaFace  | 0.9950 | 0.9401 | 0.9150  | 0.9538  | 0.9617 | 0.9531    |


## Mixed Quality Scenario (IJBB, IJBC Dataset)

For masked IJBB, IJBC validation, refer to 
```
cd validation_mixed
bash eval_ijb.sh
```
A: TAR@FAR = 0.0001%
B: TAR@FAR = 0.01%
| Arch | Dataset    | Method      | $IJBB_A$ | $IJBB_B$ | $IJBC_A$ | $IJBC_B$ | 
|------|------------|-------------|--------------------|--------------------|--------------------|--------------------|
| R50  | CASIA-WebFace       | AdaFace | NA              | 92.95              | NA              | 94.92              |
| R50  | CASIA-WebFace (masked)  | AdaFace | NA              | 92.39              | NA              | 92.85              |
| R50  | MS1MV2  | AdaFace | 94.82              | 97.59              | 96.27              | 98.40              |
| R50  | MS1MV2 (masked)     | AdaFace | 93.06              | 97.67              | 95.02              | 98.31              |

## Mixed Quality Scenario (IJBB, IJBC Masked Dataset)

For masked IJBB, IJBC validation, refer to 
```
cd validation_mixed
bash eval_ijb.sh
```

A: TAR@FAR = 0.0001%
B: TAR@FAR = 0.01%
| Arch | Dataset    | Method      | $IJBB_A$ | $IJBB_B$ | $IJBC_A$ | $IJBC_B$ | 
|------|------------|-------------|--------------------|--------------------|--------------------|--------------------|
| R50  | CASIA-WebFace       | AdaFace | NA              | 50.61              | NA              | 42.48              |
| R50  | CASIA-WebFace (masked)  | AdaFace | NA              | 52.74              | NA              | 50.12              |
| R50  | MS1MV2  | AdaFace | 90.00              | 96.61              | 92.52              | 97.44              |
| R50  | MS1MV2 (masked)     | AdaFace | 90.88              | 97.34              | 93.04              | 97.99              |

## Average Result

For average result, refer to 

A: TAR@FAR = 0.0001%
B: TAR@FAR = 0.01%
| Arch | Dataset    | Method      | High Quality | $IJBB_A$ | $IJBB_B$ | $IJBC_A$ | $IJBC_B$ | 
|------|------------|-------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| R50  | CASIA-WebFace       | AdaFace | 92.34              | NA              | 72.40              | NA              | 68.70              |
| R50  | CASIA-WebFace (masked)  | AdaFace | 92.51              | NA              | 72.57              | NA              | 71.49              |
| R50  | MS1MV2  | AdaFace | 95.62              | 92.41              | 97.10              | 94.40              | 97.92              |
| R50  | MS1MV2 (masked)     | AdaFace | 95.64              | 91.97              | 97.51              | 94.03              | 98.15              |
