# now editing...
## FDFtNet: Facing Off Fake Images using Fake Detection Fine-tuning Network.
Hyeonseong Jeon, Youngoh Bang,  and Simon S. Woo.

 
## (1) Setup
### Install packages
- 'pip install -r requirements.txt'
- Download model weights
- 'bash weights/download_weights.sh'

## (2) Quick start
'''
# Model weights need to be downloaded.
python demo.py examples/real.png weights/blur_jpg_prob0.1.pth
python demo.py examples/fake.png weights/blur_jpg_prob0.1.pth
demo.py simply runs the model on a single image, and outputs the uncalibrated prediction.
'''

## (3) Dataset
The testset evaluated in the paper can be downloaded here.

'''
# Download the dataset
cd dataset
bash download_testset.sh
cd ..
'''

## (4) Train


## (5) Evaluation
