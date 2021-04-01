import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if __name__ == '__main__':
    tf.config.list_physical_devices("GPU")