# INSTALL ANACONDA BEFORE RUNNING THE SETUP
# From the public folder https://disk.yandex.ru/d/VDWjzzpifhpBTw
# download the files:
# face_inpainting/medical/CelebA-HQ-img-256-256.zip
# face_inpainting/medical/CelebA-HQ-img-256-256-labels.zip
# face_inpainting/medical/CelebA-HQ-img-256-256-masked.zip
# face_inpainting/medical/CelebA-HQ-img-256-256-merged.zip
# into the project root (the folder where this script is located)
# and run this script to setup the environment

# The models are under the face_inpainting/models folder
# download some of the models and place them under the saved_models folder

mkdir train_data
mkdir train_data/medical
mkdir train_data/medical/CelebA-HQ-img-256-256
mkdir train_data/medical/CelebA-HQ-img-256-256-labels
mkdir train_data/medical/CelebA-HQ-img-256-256-masked
mkdir train_data/medical/CelebA-HQ-img-256-256-merged

unzip CelebA-HQ-img-256-256.zip -d train_data/medical
unzip CelebA-HQ-img-256-256-labels.zip -d train_data/medical
unzip CelebA-HQ-img-256-256-masked.zip -d train_data/medical
unzip CelebA-HQ-img-256-256-merged.zip -d train_data/medical

mkdir binary_segmentation
mkdir compare
mkdir metrics
mkdir metrics/custom
mkdir metrics/custom/real
mkdir metrics/custom/generated
mkdir saved_models
mkdir face_segmentation
mkdir inpaint_real
mkdir video_capture
mkdir merged_binary_face
mkdir gan_images
mkdir video

# Add Conda Forge (Community package repository)
conda config --append channels conda-forge

# Create an environment and link it in an IDE if needed
conda create --name env python=3.7
conda activate env
conda install tensorflow=2.0.0
conda install scikit-learn
pip install face-recognition==1.3.0
pip install numpy==1.19.2
pip install tensorboard==2.0.2
pip install tensorflow-estimator==2.0.0
pip install dlib==19.22.0
pip install opencv-python
pip install scikit-image
pip install tensorflow_addons