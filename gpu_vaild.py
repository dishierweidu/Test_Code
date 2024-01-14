import tensorflow as tf
import torch

# Check if GPU is available
print("tensorflow_gpu: ",tf.test.is_gpu_available())
# print("torch_gpu: ",torch.cuda.is_available())