import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import onnx
import torch
import tensorflow as tf
import numpy as np
import onnxruntime as ort
from onnx2torch import convert

if not os.path.exists('models'):
    os.mkdir('models')

if not os.path.exists('models/tf2pytorch'):
    os.mkdir('models/tf2pytorch')

saved_model_dir = "./models/tf2pytorch"
onnx_model_path = "./models/tf2pytorch/model.onnx"

trash_pad_tf = tf.constant([-1])
trash_pad_torch = torch.tensor([-1], dtype=torch.int32)
trash_pad_onnx = trash_pad_torch.numpy()

class CustomModule(tf.Module):
    def __init__(self):
        super(CustomModule, self).__init__()
    
    @tf.function(input_signature=[tf.TensorSpec([1], tf.int32)])
    def __call__(self, x):
        return tf.raw_ops.Abs(x=x, name=None)

def build_pb():
    module = CustomModule()
    tf.saved_model.save(module, saved_model_dir)

def onnx2pytorch(onnx_model_path):
    torch_model = convert(onnx_model_path)
    return torch_model

def tf_model_test():
    graph = tf.saved_model.load(saved_model_dir)
    return graph(trash_pad_tf)

def pytorch_model_test(onnx_model_path):
    torch_model = onnx2pytorch(onnx_model_path)
    return torch_model(trash_pad_torch)

def onnx_model_test(onnx_model_path):
    ort_sess = ort.InferenceSession(onnx_model_path)
    outputs_ort = ort_sess.run(None, {'x': trash_pad_onnx})
    return outputs_ort

def tf2onnx(onnx_model_path):
    cmd = f'python3 -m tf2onnx.convert --saved-model {saved_model_dir}  --output {onnx_model_path}'
    os.system(cmd)

build_pb()
tf_res = tf_model_test()
print(tf_res)
tf2onnx(onnx_model_path)

print(onnx_model_test(onnx_model_path))
pytorch_res = pytorch_model_test(onnx_model_path)
print(pytorch_res)