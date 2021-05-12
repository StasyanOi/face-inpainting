curl -O https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
chmod +x Anaconda3-2020.07-Linux-x86_64.sh
conda create --name tf_gpu
conda activate tf_gpu
conda install tensorflow
conda install scikit-learn
conda pip install face_recognition
