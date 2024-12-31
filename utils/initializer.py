import socket
import os
import tensorflow as tf
TEMP_DIR = "tmp"
os.makedirs(TEMP_DIR, exist_ok=True)
print("Hostname: ", socket.gethostname())
try:  # for CUDA enviroment
    os.system("nvidia-smi")  # Check GPU
except:
    pass

print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print("TensorFlow Devices: ", tf.config.get_visible_devices())
print("TensorFlow GPUs:", tf.config.list_physical_devices('GPU'))

METRIC_TITLES = {  # To translate eval metric names to readable chart titles, add any new metric here
    'accuracy': 'Evaluation Accuracy',
    'loss': 'Evaluation Loss',
    'recall': 'Evaluation Recall',
    'precision': 'Evaluation Precision',
    'f1_score': 'Evaluation F1 Score',
    'auc': 'Evaluation AUC'
}
