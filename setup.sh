# INSTALL ANACONDA BEFORE RUNNING THE SETUP
# From the public folder https://disk.yandex.ru/d/VDWjzzpifhpBTw
# download the files:
# face_inpainting/medical/CelebA-HQ-img-256-256.zip
# face_inpainting/medical/CelebA-HQ-img-256-256-labels.zip
# face_inpainting/medical/CelebA-HQ-img-256-256-masked.zip
# face_inpainting/medical/CelebA-HQ-img-256-256-merged.zip
# into the project root (the folder where this script is located)
# and run this script to setup the environment

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
mkdir face_segmentation
mkdir inpaint_real
mkdir video_capture
mkdir merged_binary_face

# Add Conda Forge (Community package repository)
conda config --append channels conda-forge

# Create an environment and link it in an IDE if needed
conda create --name env python=3.7
conda activate env
conda install tensorflow
conda install scikit-learn
conda install face_recognition
pip install opencv-python
pip install scikit-image