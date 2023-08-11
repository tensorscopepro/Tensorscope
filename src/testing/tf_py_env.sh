pip install -U tf-nightly-cpu

wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/core/ops/ops.pbtxt

python tf_fuzzer.py 1> test_log 2>&1