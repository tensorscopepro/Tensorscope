import re
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import ast 
import time
import astunparse
import numpy as np
import tensorflow as tf
import string
import random
import inspect
from loguru import logger
from google.protobuf import text_format
from tensorflow.core.framework import op_def_pb2 

prefix = 'tf.raw_ops.{}'
MAX_FUZZ_ITER = 200
ST_END = 0
ST_FIX_BY_INDX = 1
ST_FIX_BY_NAME = 2
ST_RST = 3
ST_CON = 4

TypeList = [
    ('DT_INVALID', ''),
    # Data types that all computation devices are expected to be
    # capable to support.
    ('DT_FLOAT', 'tf.dtypes.float32'),
    ('DT_DOUBLE', 'tf.dtypes.float64'),
    ('DT_INT32', 'tf.dtypes.int32'),
    ('DT_UINT8', 'tf.dtypes.uint8'),
    ('DT_INT16', 'tf.dtypes.int16'),
    ('DT_INT8', 'tf.dtypes.int8'),
    ('DT_STRING', 'tf.dtypes.string'),
    ('DT_COMPLEX64', 'tf.dtypes.complex64'),  # Single-precision complex
    ('DT_INT64', 'tf.dtypes.int64'),
    ('DT_BOOL', 'tf.dtypes.bool'),
    ('DT_QINT8', 'tf.dtypes.qint8'),          # Quantized int8
    ('DT_QUINT8', 'tf.dtypes.quint8'),        # Quantized uint8
    ('DT_QINT32', 'tf.dtypes.qint32'),        # Quantized int32
    # Float32 truncated to 16 bits.  Only for cast ops.
    ('DT_BFLOAT16', 'tf.dtypes.bfloat16'),
    ('DT_QINT16', 'tf.dtypes.qint16'),        # Quantized int16
    ('DT_QUINT16', 'tf.dtypes.quint16'),      # Quantized uint16
    ('DT_UINT16', 'tf.dtypes.uint16'),
    ('DT_COMPLEX128', 'tf.dtypes.complex128'),  # Double-precision complex
    ('DT_HALF', 'tf.dtypes.half'),
    ('DT_RESOURCE', 'tf.dtypes.resource'),
    ('DT_VARIANT', 'tf.dtypes.variant'),      # Arbitrary C++ data types
    ('DT_UINT32', 'tf.dtypes.uint32'),
    ('DT_UINT64', 'tf.dtypes.uint64'),
]

def parser(pb_file, debug=False, debug_target=''):
    with open(pb_file, 'rb') as f:
        ops_desc = f.read()
    buffer = op_def_pb2.OpList()
    text_format.Parse(ops_desc, buffer)

    OP_DICT = {}

    for op in buffer.op:
        if debug and op.name != debug_target:
            continue
        op_attr = {}
        for attr in op.attr:
            if attr.type == 'string':
                allows = attr.allowed_values.list.s
                types = allows if allows else [b'']
                op_attr[attr.name] = {
                    'type_name' : types,
                    'is_str' : True,
                    'is_type' : False,
                    'min' : None,
                    'default' : attr.default_value.s
                }
            elif attr.type == 'type':
                allows = attr.allowed_values.list.type
                types = [TypeList[i][0] for i in allows] \
                        if allows else ['DT_INT32', 'DT_FLOAT']
                op_attr[attr.name] = {
                    'type_name' : types,
                    'is_str' : False,
                    'is_type' : True,
                    'min' : None,
                    'default' : attr.default_value
                }
            elif attr.has_minimum:
                op_attr[attr.name] = {
                    'type_name' : [attr.type],
                    'is_str' : False,
                    'is_type' : False,
                    'min' : attr.minimum,
                    'default' : attr.default_value
                }
            else:
                op_attr[attr.name] = {
                    'type_name' : [attr.type],
                    'is_str' : False,
                    'is_type' : False,
                    'min' : None,
                    'default' : attr.default_value
                }

        input_arg = {}
        output_arg = []

        for item in op.input_arg:
            if item.type:
                input_arg[item.name] = {'type': {'type_name': [
                    TypeList[item.type][0]]}, 'num_attr': None}
            elif item.type_attr:
                input_arg[item.name] = {
                    'type': op_attr[item.type_attr], 'num_attr': None}
            elif item.type_list_attr:
                input_arg[item.name] = {
                    'type': op_attr[item.type_list_attr], 'num_attr': None}
            if item.number_attr:
                input_arg[item.name]['num_attr'] = op_attr[item.number_attr]

        for item in op.output_arg:
            if item.type:
                output_arg.append((item.name, item.type))
            if item.type_attr:
                output_arg.append((item.name, op_attr[item.type_attr]))

        OP_DICT[op.name] = {
            'input_arg': input_arg,
            'output_arg': output_arg,
            'op_attr': op_attr,
        }
        ############ debug ###############
        if debug:
            print(input_arg)
            return OP_DICT
        ##################################
    return OP_DICT

def random_printable(len):
    candidate = list(string.printable)[:-7]
    res = ''.join(random.sample(candidate, len)).replace('"', '')
    return f'"{res}"'

def random_tensor(dtype):

    rand_dim = random.choice([0, 1, 2, 4, 8])
    rand_shape = [random.choice([0, 1, 2, 4, 8]) for i in range(rand_dim)]

    if 'string' in dtype:
        return random_printable(random.randint(5, 25))
    if 'bool' in dtype:
        return random.choose([True, False])
    if 'float' in dtype:
        return f"tf.random.uniform({rand_shape}, dtype=tf.dtypes.float32, maxval=100000000)"
    if 'half' in dtype:
        return f"tf.random.uniform({rand_shape}, dtype=tf.dtypes.half, maxval=1000)"
    if 'double' in dtype:
        return f"tf.random.uniform({rand_shape}, dtype=tf.dtypes.double, maxval=1000000000)"
    if 'int8' in dtype:
        return f"tf.random.uniform({rand_shape}, dtype=tf.dtypes.int32, minval=-50, maxval=200)"
    if 'int16' in dtype:
        return f"tf.random.uniform({rand_shape}, dtype=tf.dtypes.int32, minval=-10000, maxval=60000)"
    if 'int32' in dtype:
        return f"tf.random.uniform({rand_shape}, dtype=tf.dtypes.int32, minval=-100000, maxval=1000000)"
    if 'int64' in dtype:
        return f"tf.random.uniform({rand_shape}, dtype=tf.dtypes.int64, minval=-100000000, maxval=1000000000)"
    if 'complex64' in dtype:
        return f"tf.cast(tf.random.uniform({rand_shape}, dtype=tf.dtypes.float32, maxval=60000), dtype=tf.complex64)"
    elif 'complex128' in dtype:
        return f"tf.cast(tf.random.uniform({rand_shape}, dtype=tf.dtypes.float32, maxval=60000), dtype=tf.complex128)"
    elif dtype == 'shape':
        # return "tf.TensorShape(None)"
        tmp_rand_shape = [random.choice([0, 1, -2, 4, -8]) for i in range(rand_dim)]
        return f"tf.constant({tmp_rand_shape})"
    elif dtype == 'type':
        return "tf.dtypes.float32"  # TODO
    elif 'variant' in dtype:
        return "tf.data.experimental.to_variant(tf.data.Dataset.from_tensor_slices([1, 2, 3]))"
    else:
        return f"tf.random.uniform({rand_shape}, dtype=tf.dtypes.float32, maxval=100000)"

def random_const(dtype):
    if 'int' in dtype:
        return random.randint(-30, 255)
    if ('double' in dtype) or ('float' in dtype):
        return random.uniform(0.0, 15.5)
    if 'string' in dtype:
        return random_printable(random.randint(5, 25))
    if 'half' in dtype:
        return "tf.constant(random.uniform(0.0, 15.5), dtype=tf.half)"
    if 'bool' in dtype:
        return random.choice([True, False])
    if 'tensor' in dtype:
        return random_tensor(f'dt_int{random.choice([8, 16, 32])}')
    if 'shape' in dtype:
        return "tf.TensorShape(None)"
    if 'type' in dtype:
        return "tf.dtypes.float32"

    print("[-] Missing type in random_const!")
    return None

def match_rules(desc):
    # print("================", desc)

    # cannot compute Add as input #1(zero-based) was expected to be a int32 tensor but is a float tensor [Op:Add]
    invalid_arg_pattern = re.compile(
        r'\#(\d).* was expected to be a (.*) tensor but is a (.*) tensor')
    invalid_arg_pattern_res = invalid_arg_pattern.search(desc)

    err_msg1 = re.compile(r'(\w+) must be [a]?[ ]?(\w+)')
    err_msg1_res = err_msg1.search(desc)

    wrong_type_pattern = re.compile(r'Expected () for argument')

    missing_arg_pattern = re.compile(
        r"missing 1 required positional argument: '(.*)'")

    # must be greater than

    if "not support eager execution" in desc:
        return (ST_END, None)
    elif "It has been removed in version" in desc:
        return (ST_END, None)
    # elif "not find device for node" in desc: # maybe special
    #    return (ST_END, None)
    elif invalid_arg_pattern_res:
        return (ST_FIX_BY_INDX, (invalid_arg_pattern_res.group(1),
                                 invalid_arg_pattern_res.group(2),
                                 invalid_arg_pattern_res.group(3)))
    elif err_msg1_res:
        return (ST_FIX_BY_NAME, (err_msg1_res.group(1),
                                 err_msg1_res.group(2)))
    else:
        return (ST_RST, None)

def modify_expr_by_indx(old_expr, req, guide):
    place, need_dtype, org_dtype = req
    astree = ast.parse(old_expr, mode='eval')
    change = random_tensor(need_dtype)
    astree.body.keywords[int(place)].value = ast.parse(change).body[0].value
    new_expr = astunparse.unparse(astree)
    return new_expr

def modify_expr_by_name(old_expr, req, guide):
    print(old_expr)
    exit()

def gen_fuzz_input(arg_details, is_input=False, is_scalar=False):
    arg_types = arg_details['type'] if is_input else arg_details
    dtype = random.choice(arg_types['type_name']).lower()
    dtype = dtype.decode() if isinstance(dtype, bytes) else dtype

    if is_input and arg_details['num_attr']:
        minimum = arg_details['num_attr']['min']
        return f"{random.randint(minimum, 50)} * [{random_tensor(dtype)}]"

    if is_input == False and arg_types['is_type']:
        # is type, generate type
        for raw_type, tf_type in TypeList:
            if raw_type.lower() == dtype:
                return tf_type
        logger.warning('Missing type in typelist?')
        return None

    if is_input == False and arg_types['is_str']:
        # is str, generate str
        if dtype != '':
            return f'"{dtype}"'
        elif arg_types['default'] != b'':
            default_val = arg_types['default']
            return f'"{default_val}"'
        else:
            return random_printable(random.randint(3, 10))
    
    if is_input == False or is_scalar:
        # not type not str, generate random const
        if dtype == 'shape':
            return random_tensor(dtype)
        if 'list' in dtype:
            p = re.compile(r'list\((.*)\)')
            base_type = p.search(dtype).group(1)
            return f"{arg_types['minimum']} * [{random_const(base_type)}]"
        return random_const(dtype)

    return random_tensor(dtype)

def fuzz_single(op, op_info):
    info = inspect.signature(eval(prefix.format(op)))
    logger.info(f'Start fuzzing {op} with info {info}')

    state = ST_RST

    for _ in range(MAX_FUZZ_ITER):
        fuzz_data = ''
        if state == ST_RST:
            for arg in info.parameters:
                if arg in op_info['input_arg']:
                    arg_details = op_info['input_arg'][arg]
                    generated_arg_input = gen_fuzz_input(arg_details, is_input=True)
                    fuzz_data += f'{arg}={generated_arg_input},'

                elif arg in op_info['op_attr']:
                    arg_details = op_info['op_attr'][arg]
                    generated_arg_input = gen_fuzz_input(arg_details)
                    fuzz_data += f'{arg}={generated_arg_input},'
            code = prefix.format(f'{op}({fuzz_data})')
        logger.info(f'[Round {_+1}] Generated code: \n{code}')
        state = ST_RST
        try:
            exec(code)
        except (tf.errors.UnimplementedError,
                tf.errors.NotFoundError,
                RuntimeError,
                TypeError,
                tf.errors.InvalidArgumentError,
                tf.errors.InternalError,
                UnicodeDecodeError,
                tf.errors.FailedPreconditionError,
                tf.errors.OutOfRangeError) as err:
            logger.warning(f'EROOR: {err}')
            state, req = match_rules(str(err))
            if state == ST_END:
                break
            elif state == ST_FIX_BY_INDX:
                code = modify_expr_by_indx(code, req, op_info)
            elif state == ST_FIX_BY_NAME:
                # code = modify_expr_by_name(code, req, op_info)
                #TODO
                state = ST_RST
                pass
            elif state == ST_RST:
                pass

        
known = ['Abort', 'AudioSummaryV2', 'CollectiveInitializeCommunicator', 'Diag']

OP_DICT = parser('ops.pbtxt', debug=False, debug_target='SetSize')
flag = 0
for idx, (op, op_info) in enumerate(OP_DICT.items()):
    if op == 'RaggedTensorToVariantGradient':
        flag = 1
        continue
    if flag == 0:
        continue
    try:
        fuzz_single(op, op_info)
    except:
        pass
    # if op == 'CollectiveInitializeCommunicator':
    #     fuzz_single(op, op_info)
    #     exit()