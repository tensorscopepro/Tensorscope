import os
import re
import sys

import ast
import astpretty
import argparse
import astunparse

import numpy as np
import tensorflow as tf
import onnxruntime as rt

saved_tf_model_dir = "onnx_test/tf_model/"
saved_onnx_model_path = "onnx_test/onnx_model/"
log_files_path = "onnx_test/onnx_fuzz_log/"

if(not os.path.exists(saved_tf_model_dir)):
    os.makedirs(saved_tf_model_dir)

if(not os.path.exists(saved_onnx_model_path)):
    os.makedirs(saved_onnx_model_path)

if(not os.path.exists(log_files_path)):
    os.makedirs(log_files_path)


class OnnxModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = rt.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name
 
    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name
 
    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        i = 0
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy[i]
            i += 1
        return input_feed
 
    def forward(self, numpy_list):
        '''
        # image_numpy = image.transpose(2, 0, 1)
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy:
        # :return:
        '''
 
        input_feed = self.get_input_feed(self.input_name, numpy_list)
        # scores = self.onnx_session.run(self.output_name[0], input_feed=input_feed)
        # print(input_feed)
        output = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return output

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
        print(arg_name, arg_value_raw, arg_value_str)
        if type(arg_value_raw) == type(ast.Constant()):
            data = eval(arg_value_str)
            dtype = type(data).__name__
        elif type(arg_value_raw) == type(ast.Attribute()):
            data = arg_value_str
            dtype = type(arg_value_str).__name__
        elif 'dtype' in arg_value_str:
            dtype = re.search('dtype=(\w+.\w+.\w+)', arg_value_str).group(1)
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

def build_tf_model(arg_info, saved_tf_model_dir):

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
    
    # https://stackoverflow.com/questions/60974077/how-to-save-keras-model-as-frozen-graph
    full_model = tf.function(lambda x: module(*x))
    full_model = full_model.get_concrete_function(signature)
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./onnx_test/tf_model",
                      name=f"{func}_frozen_graph.pb",
                      as_text=False)
    tf.saved_model.save(module, saved_tf_model_dir)

def tf_model_test(saved_tf_model_dir, test_args):
    graph = tf.saved_model.load(saved_tf_model_dir)
    return graph(*test_args)

def tf2onnx(saved_tf_model_dir, saved_onnx_model_path):
    cmd = f'python -m tf2onnx.convert --saved-model {saved_tf_model_dir}  --output {saved_onnx_model_path} --opset 17'
    os.system(cmd)

def onnx_model_test(onnx_model_path, test_args):
    model = OnnxModel(onnx_model_path)
    return model.forward(test_args)

def save_test_data(api, test_input):

    data_dir = saved_tf_model_dir + '/' + api + '/'
    if(not os.path.exists(data_dir)):
        os.makedirs(data_dir)
    for para_name, data, _ in test_input:
        np.save(data_dir + para_name, data)

def main():

    # cmd = sys.argv[1]

    cmd = "tf.raw_ops.AdjustContrastv2(images=tf.random.uniform([64, 64, 3, 2], dtype=tf.dtypes.float32, maxval=255),contrast_factor=tf.random.uniform([], dtype=tf.dtypes.float32, maxval=1),)"
    
    # cmd = "tf.raw_ops.AdjustContrastv2(images=2*tf.ones([64, 64, 3, 2], dtype=tf.dtypes.float32, ), contrast_factor=tf.constant(1.5,dtype=tf.dtypes.float32,))"
    
    # cmd = "tf.raw_ops.AdjustContrastv2(images=5*tf.eye(64, num_columns=64, batch_shape=(3, 2), dtype=tf.dtypes.float32, ), contrast_factor=tf.constant(1.5,dtype=tf.dtypes.float32,))"

    log_file = log_files_path + "onnx_test.log"
    fw_log = open(log_file,"a+")

    print(f"{cmd}",file=fw_log)

    arg_info = clean_data(cmd)
    print(arg_info)
    try:
        # tf.raw_ops test

        fuc_name = arg_info[0]
        raw_test_args = []
        for i in range(1, len(arg_info)):
            arg_name, data, dtype = arg_info[i]
            if 'tf.' in dtype:
                raw_test_args.append(f"{arg_name}=arg_info[{i}][1]" )

        raw_cmd = fuc_name + "(" + ",".join(raw_test_args)+")"

        print("raw_cmd",file=fw_log)
        print(raw_cmd,file=fw_log)

        raw_res = eval(raw_cmd)
        
        save_test_data(arg_info[0], arg_info[1:])
        
    except (Exception, ArithmeticError) as e:
        print("tf.raw_ops test error",file=fw_log)
        print(e, file=fw_log)
        print("\n\n", file=fw_log)
        fw_log.close()
        exit()
    else:
        pass

    try:
        # tf_model test

        build_tf_model(arg_info, saved_tf_model_dir)

        tf_test_args = []
        for i in range(1, len(arg_info)):
            arg_name, data, dtype = arg_info[i]
            if 'tf.' in dtype:
                tf_test_args.append(data)

        tf_res = tf_model_test(saved_tf_model_dir, tf_test_args)

    except (Exception, ArithmeticError) as e:
        print("tf_model test error",file=fw_log)
        print(e, file=fw_log)
        print("\n\n", file=fw_log)
        fw_log.close()
        exit()
    else:
        pass

    try:
        # onnx_model test

        onnx_test_args = []
        for i in range(1, len(arg_info)):
            arg_name, data, dtype = arg_info[i]
            if 'tf.' in dtype:
                if isinstance(data.numpy(),np.ndarray):
                    onnx_test_args.append(data.numpy())
                else:
                    # In ONNX input, Scalar is not run, so it needs to be converted into Tensor.
                    onnx_test_args.append(np.array(data.numpy()))
            
        onnx_path = saved_onnx_model_path + f"{arg_info[0]}_model.onnx"

        tf2onnx(saved_tf_model_dir, onnx_path)

        onnx_res = onnx_model_test(onnx_path, onnx_test_args)

    except (Exception, ArithmeticError) as e:
        print("onnx_model test error",file=fw_log)
        print(e, file=fw_log)
        print("\n\n", file=fw_log)
        fw_log.close()
        exit()
    else:
        pass

    print(f"tf_res : {raw_res.numpy()}", file=fw_log)
    print(f"onnx_res : {onnx_res}", file=fw_log)
    print(f"diff : {(tf_res.numpy() - onnx_res)}", file=fw_log)
    #print(f"raw_res - tf_res : {(raw_res.numpy() - tf_res.numpy()).sum()}", file=fw_log)
    #print(f"tf_res - onnx_res : {(tf_res.numpy() - onnx_res).sum()} \n\n", file=fw_log)
    
    np.testing.assert_allclose(tf_res.numpy(), np.squeeze(onnx_res), rtol=1e-4, atol=1e-4)
    
    fw_log.close()


if __name__ == '__main__':
    main()

    '''
    tested on: 
    python onnx_op_test.py "tf.raw_ops.Abs(x=tf.random.uniform([2], dtype=tf.dtypes.int32, minval=-100000, maxval=1000000),)"
    '''