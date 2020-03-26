# now editing... 
Upload is being delayed by Coronavirus in our country. Thank you for your interest and Sorry for being delayed.

## FDFtNet: Facing Off Fake Images using Fake Detection Fine-tuning Network.
Hyeonseong Jeon, Youngoh Bang,  and Simon S. Woo.
 
## (1) Setup
### Install packages
- `pip install -r requirements.txt`

## (2) Datasets
The dataset in the paper can be downloaded here.
* Deepfake: https://github.com/ondyari/FaceForensics
* Face2Face: https://github.com/ondyari/FaceForensics
* PGGAN dataset is not disclosed at the request of the dataset source.

Original dataset of the Deepfake, Face2Face can also be downloaded from FaceForensics
### Preprocessing
Will be uploaded soon...
```
# Crop the face parts to image from the videos by frame
cd dataset
python preprocess_dataset.py -d youtube
```
```
python preprocess_dataset.py
 -d <dataset type, e.g., youtube, Deepfakes, Face2Face>
```

## (3) Models
* SqueezeNet
* ResNetV2
* Xception
* ShallowNetV3 is not disclosed at the request of the model source.

## (4) Quick-start


## (5) Pre-train


## (6) Fine-tune


## (7) Evaluate
