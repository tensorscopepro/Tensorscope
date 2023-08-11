# bazel test -c opt --config asan //tensorflow/lite/kernels:combined_all_kernel_tests

import os
import tensorflow as tf # type: ignore
import numpy as np

saved_model_dir = "./model"

m = 200
k = 256
n = 128

a_shape = [m, k]
b_shape = [k, n]

np.random.seed(0)

input_np = np.random.uniform(low=0.0, high=1.0, size=a_shape).astype("float32")
kernel_np = np.random.uniform(low=0.0, high=1.0, size=b_shape).astype("float32")

def build_pb_from_tf1():
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    input_np = np.random.uniform(low=0.0, high=1.0, size=a_shape).astype("float32")
    kernel_np = np.random.uniform(low=0.0, high=1.0, size=b_shape).astype("float32")

    pld1 = tf.compat.v1.placeholder(dtype="float32", shape=a_shape, name="input1")
    kernel = tf.constant(kernel_np, dtype="float32")
    feed_dict = {pld1: input_np}

    result_tf = tf.raw_ops.MatMul(a=pld1, b=kernel, transpose_a=False, transpose_b=False)

    with tf.compat.v1.Session() as sess:
        results = sess.run(result_tf, feed_dict=feed_dict)
        print("results:", results)

    dump_model_name = os.path.join(saved_model_dir, "matmul_graph.pb")

    graph = tf.compat.v1.get_default_graph()
    graph_def = graph.as_graph_def()
    with tf.io.gfile.GFile(dump_model_name, "wb") as f:
        f.write(graph_def.SerializeToString())

class CustomModule(tf.Module):
    
    def __init__(self):
        super(CustomModule, self).__init__()
        self.kernel = kernel_np
        
    @tf.function(input_signature=[tf.TensorSpec(a_shape, tf.float32)])
    def __call__(self, x):
        print('Tracing with', x)
        return tf.raw_ops.MatMul(a=x, b=self.kernel, transpose_a=False, transpose_b=False)
    
    @tf.function(input_signature=[tf.TensorSpec(a_shape, tf.float32)])
    def mutate(self, new_kernel):
        self.kernel = new_kernel

def build_pb_from_saved_model():
    module = CustomModule()
    # module(tf.constant(0.)) lazy get_concrete_function ValueError: Found zero restored functions for caller function.
    tf.saved_model.save(module, saved_model_dir)

def convert_tflite_from_saved_model(tflite_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

def saved_model_test():

    graph = tf.saved_model.load(saved_model_dir)
    print(graph(input_np).numpy())

def tflite_model_test(tflite_path):

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    print(interpreter.get_output_details())
    print(interpreter.get_input_details())
    output = interpreter.get_output_details()[0]
    input = interpreter.get_input_details()[0]
    # input_np = np.random.uniform(low=0.0, high=1.0, size=a_shape).astype("float32")
    interpreter.set_tensor(input['index'], input_np)
    interpreter.invoke()
    print(interpreter.get_tensor(output['index']))

build_pb_from_saved_model()
saved_model_test()
tflite_path = "model.tflite"
convert_tflite_from_saved_model(tflite_path)
tflite_model_test(tflite_path)


# ./saved_model_cli show --dir /host_dir/model/ --tag_set serve --signature_def serving_default
# ./saved_model_cli run --dir /host_dir/model/ --tag_set serve --signature_def serving_default --input_exprs x=3