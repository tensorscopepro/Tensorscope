#### TF
weggli -X '{OP_REQUIRES(_, _, _);}' tensorflow/tensorflow/core/kernels/ > tf.op_requires.constraints
# weggli -X -B 0 -A 0 '{OP_REQUIRES($a, _, InvalidArgument(_));}' tensorflow/tensorflow/core/kernels/

semgrep --config=py.Error.yaml tensorflow/tensorflow/python/ops/ -o tf.py.value_type_Error.constraints

#### PT
# function names strat from `TORCH_CHECK`
weggli -X -R func=TORCH_CHECK '{$func(_);}' pytorch/aten/src/ATen/native/ > extract_constraints/pt.torchcheck.constraints 

# \x1b\[\d+m is the color code
semgrep --config=py.Error.yaml pytorch/torch/ -o extract_constraints/pt.py.value_type_Error.constraints

#### MS
weggli -X '{MS_EXCEPTION(_);}' mindspore/mindspore/core/ops > extract_constraints/ms.cc.constraints

semgrep --config=extract_constraints/py.Error.yaml mindspore/mindspore/python/mindspore/ops -o extract_constraints/ms.py.constraints

#### Paddle
semgrep --config=extract_constraints/cc.Paddle_Assert.yaml Paddle/paddle/fluid/operators -o extract_constraints/paddle.cc.constraints

semgrep --config=extract_constraints/py.Error.yaml Paddle/python/paddle/nn -o extract_constraints/paddle.py.constraints

#### ORT
semgrep --config=extract_constraints/cc.ORT_Assert.yaml onnxruntime/onnxruntime/contrib_ops -o extract_constraints/ort.cc.constraints

semgrep --config=extract_constraints/py.Error.yaml  -o extract_constraints/ort.py.constraints 