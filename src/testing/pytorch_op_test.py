import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import re
import ast
import onnx
import torch
import argparse
import astunparse
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

trash_pad_tf = tf.constant([1])
trash_pad_torch = torch.tensor([1], dtype=torch.int32)
trash_pad_onnx = trash_pad_torch.numpy()


def clean_data(cmd):
    print(f'[+] Current command: {cmd}')
    func = cmd[:cmd.find('(')]
    print(f'[+] Extracted function: {func}')
    astree = ast.parse(cmd, mode='eval')
    arg_num = len(astree.body.keywords)
    arg_info = []
    for i in range(arg_num):
        arg_name = astree.body.keywords[i].arg
        arg_value_raw = astree.body.keywords[i].value
        arg_value_str = astunparse.unparse(arg_value_raw).strip()
        if type(arg_value_raw) == type(ast.Constant()):
            data = eval(arg_value_str)
            dtype = type(data).__name__
        elif type(arg_value_raw) == type(ast.Attribute()):
            data = arg_value_str
            dtype = type(arg_value_str).__name__
        elif 'dtype' in arg_value_str:
            dtype = re.search('dtype=(.*?),', arg_value_str).group(1)
            data = eval(arg_value_str)
        arg_info.append((arg_name, data, dtype))
    arg_info = [func] + arg_info
    return arg_info

def gen_signature(arg_info):
    signatures = []
    for args in arg_info:
        arg_name, data, dtype = args
        if 'tf.' in dtype:
            signature = tf.TensorSpec(list(data.shape), dtype=eval(dtype))
            signatures.append(signature)
    return signatures

def build_pb(arg_info):
    func = arg_info[0]
    arg_info = arg_info[1:]
    signature = gen_signature(arg_info)
    print(f'[+] Generated signature: {signature}')
    cmd_model = func + '('
    for i in range(len(arg_info)):
        arg_name, data, dtype = arg_info[i]
        if 'tf.' in dtype:
            cmd_model += f'{arg_name}=args[{i}],'
        else:
            cmd_model += f'{arg_name}={str(data)},'
    cmd_model += ')'

    class CustomModule(tf.Module):
        def __init__(self):
            super(CustomModule, self).__init__()

        @tf.function(input_signature=signature)
        def __call__(self, *args):
            return eval(cmd_model)

    module = CustomModule()
        
    full_model = tf.function(lambda x: module(*x))
    full_model = full_model.get_concrete_function(signature)
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./onnx_test/tf_model",
                      name=f"{func}_frozen_graph.pb",
                      as_text=False)
    
    tf.saved_model.save(module, saved_model_dir)

def onnx2pytorch(onnx_model_path):
    torch_model = convert(onnx_model_path)
    return torch_model

def tf_model_test(test_args):
    graph = tf.saved_model.load(saved_model_dir)
    return graph(*test_args)

def pytorch_model_test(onnx_model_path, test_args):
    torch_model = onnx2pytorch(onnx_model_path)
    torch.save(torch_model, saved_model_dir+f'/pt_model.pth')
    # torch_model.eval()
    for param_tensor in torch_model.state_dict():
        print("[#]", param_tensor, "\t", torch_model.state_dict()[param_tensor].size())
    print(torch_model.graph)
    return torch_model(*test_args)

def onnx_model_test(onnx_model_path):
    ort_sess = ort.InferenceSession(onnx_model_path)
    outputs_ort = ort_sess.run(None, {'x': trash_pad_onnx})
    return outputs_ort

def tf2onnx(onnx_model_path):
    cmd = f'python3 -m tf2onnx.convert --saved-model {saved_model_dir}  --output {onnx_model_path}'
    os.system(cmd)

def main(args):
    cmd = args.cmd
    arg_info = clean_data(cmd)
    build_pb(arg_info)
    tf_test_args = []
    for i in range(1, len(arg_info)):
        arg_name, data, dtype = arg_info[i]
        if 'tf.' in dtype:
            tf_test_args.append(data)
    tf_res = tf_model_test(tf_test_args)
    print(tf_res)
    tf2onnx(onnx_model_path)

    torch_test_args = []
    for tf_arg in tf_test_args:
        try:
            tmp = torch.Tensor(tf_arg.numpy())
        except TypeError:
            tmp = torch.Tensor([tf_arg.numpy()])
        torch_test_args.append(tmp)

    pytorch_res = pytorch_model_test(onnx_model_path, torch_test_args)
    print(pytorch_res)
    
    np.testing.assert_allclose(tf_res.numpy(), pytorch_res, rtol=1e-4, atol=1e-4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmd', help='cmd we need to test', type=str)
    args = parser.parse_args()
    main(args)
    '''
    tested on: 
    python3 pytorch_op_test.py --cmd="tf.raw_ops.Abs(x=tf.random.uniform([2], dtype=tf.dtypes.int32, minval=-100000, maxval=1000000),)"
    python3 pytorch_op_test.py --cmd="tf.raw_ops.AdjustContrastv2(images=tf.random.uniform([0, 2, 1, 8, 2, 4, 4, 2], dtype=tf.dtypes.float32, maxval=100000000),contrast_factor=tf.random.uniform([], dtype=tf.dtypes.float32, maxval=100000000),)"
    '''