import socket
import os
import tensorflow as tf
TEMP_DIR = "tmp"

print("Hostname: ", socket.gethostname())
try:  # for CUDA enviroment
    os.system("nvidia-smi")  # Check GPU
except:
    pass

print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print("TF All Devices: ", tf.config.get_visible_devices())
print("TF GPUs:", tf.config.list_physical_devices('GPU'))
