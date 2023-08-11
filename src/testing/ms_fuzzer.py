import os
import re
import sys
import random
import string
import numpy as np
import mindspore as ms
from loguru import logger as mylogger

MAX_FUZZ_ITER = 20000

def random_printable(len):
    candidate = list(string.printable)[:-7]
    res = ''.join(random.sample(candidate, len)).replace('"', '')
    return f'"{res}"'

def random_tensor(dtype):

    rand_dim = random.choice([0, 1, 2, 4, 8])
    rand_shape = [random.choice([0, 1, 2, 4, 8]) for i in range(rand_dim)]

    if 'string' in dtype:
        return random_printable(random.randint(5, 10))
    elif 'bool' in dtype:
        return random.choice(["True", "False"])
    elif 'int8' in dtype:
        lb = -128
        ub = 127
    elif 'uint8' in dtype:
        lb = 0
        ub = 255
    elif 'int16' in dtype:
        lb = -32768
        ub = 32767
    elif 'uint16' in dtype:
        lb = 0
        ub = 65536
    elif 'int32' in dtype:
        lb = -2147483648
        ub = 2147483647
    elif 'uint32' in dtype:
        lb = 0
        ub = 4294967295
    elif 'int64' in dtype:
        lb = -9223372036854775808
        ub = 9223372036854775807
    elif 'uint64' in dtype:
        lb = 0
        ub = 18446744073709551615   
    elif 'float16' in dtype:
        lb = -555
        ub = 1000
    elif 'float32' in dtype:
        lb = -6666666
        ub = 100000000
    elif 'float64' in dtype:
        lb = -6666666
        ub = 100000000  
    elif 'complex64' in dtype:
        lb = -10000
        ub = 60000 
    elif 'complex128' in dtype:
        lb = -10000
        ub = 60000
    else:
        print(f'ERROR: Does not have this type {dtype}!')
    return f"ms.Tensor(np.random.uniform({lb}, {ub}, {rand_shape}).astype(np.{dtype}))" 

def gen_fuzz_input(args, dtypes):
    res = ''
    arg_dtypes = random.choice(dtypes)
    for i in range(len(args)):
        arg_info = args[i]
        arg = arg_info['name']
        # res += f'{arg}='
        dtype = arg_dtypes[i][0]
        arg_value = random_tensor(dtype)
        res += arg_value
        res += ','
    return res

def fuzz_single(op):
    exec(f'import {op}')
    op_info = eval(f'{op}.{op}_op_info')
    mylogger.info(f'Start fuzzing {op} with info {op_info}')

    name = op_info['op_name']
    args = op_info['inputs']
    dtypes = op_info['dtype_format']

    for _ in range(MAX_FUZZ_ITER):
        cmd = f'{name}()('
        cmd += gen_fuzz_input(args, dtypes)
        cmd += ')'
        
        mylogger.info(f'[Round {_+1}] Generated code: \n{cmd}')
        try:
            exec(cmd)
            print('wow')
        except Exception as err:
            mylogger.warning(f'EROOR: {err}')


if __name__ == '__main__':
    import_dir = '/home/xxx/.local/lib/python3.8/site-packages/mindspore/ops/operations'
    for file in os.listdir(import_dir):
        x = file.replace('.py', '')
        if not x.startswith('_'):
            exec(f'from mindspore.ops.operations.{x} import *')
    fuzz_dir = '/home/xxx/.local/lib/python3.8/site-packages/mindspore/ops/_op_impl/aicpu'
    sys.path.append(fuzz_dir)
    ops = [x.replace('.py', '') for x in os.listdir(fuzz_dir)]
    ops.remove('__init__')
    flag = 0
    # print(ops)
    # exit()
    for op in ops:
        if op == 'linspace':
            flag = 1
            continue
        if flag:
            fuzz_single(op)
