# Download the installer for Anaconda environment manager and install Conda
curl -O https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
chmod +x Anaconda3-2020.07-Linux-x86_64.sh
./Anaconda3-2020.07-Linux-x86_64.sh

# Add Conda Forge (Community package repository)
conda config --append channels conda-forge

# Create an environment and link it in an IDE if needed
conda create --name env python=3.7
conda activate env
conda install tensorflow
conda install scikit-learn
conda install face_recognition
pip install opencv-python
