'''
modified by Hanjiang Hu for projection VNN, May 2023
based on
random test generation to find unsat systems
Stanley Bak, Feb 2021
'''

import time
import sys
import numpy as np

import onnx

from util import predict_with_onnxruntime, remove_unused_initializers
from vnnlib import read_vnnlib_simple, get_io_nodes


def run_tests(onnx_filename, vnnlib_filename, num_trials, flatten_order='C'):
    '''execute the model and its conversion as a sanity check

    returns string to print to output file
    '''

    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model, full_check=True)
    onnx_model = remove_unused_initializers(onnx_model)

    inp, out, inp_dtype, sess = get_io_nodes(onnx_model, onnx_path=onnx_filename if "cnn" in onnx_filename else None)

    inp_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in inp.type.tensor_type.shape.dim)
    out_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in out.type.tensor_type.shape.dim)

    # print(f"inp_shape: {inp_shape}")
    # print(f"out_shape: {out_shape}")

    num_inputs = 1
    num_outputs = 1

    for n in inp_shape:
        num_inputs *= n

    for n in out_shape:
        num_outputs *= n

    print(f"Testing onnx model with {num_inputs} inputs and {num_outputs} outputs")

    start = time.perf_counter()
    box_spec_list = read_vnnlib_simple(vnnlib_filename, num_inputs, num_outputs)
    diff = time.perf_counter() - start
    print(f"Parse time: {round(diff, 3)} sec")

    start = time.perf_counter()
    # inp, _ = get_io_nodes(onnx_model)
    inp_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in inp.type.tensor_type.shape.dim)
    res = 'unknown'

    for trial in range(num_trials):

        index = np.random.randint(len(box_spec_list))
        box_spec = box_spec_list[index]
        input_box = box_spec[0]
        spec_list = box_spec[1]

        input_list = []

        for lb, ub in input_box:
            r = np.random.random()

            input_list.append(lb + (ub - lb) * r)

        random_input = np.array(input_list, dtype=inp_dtype)
        random_input = random_input.reshape(inp_shape, order=flatten_order)  # check if reshape order is correct
        assert random_input.shape == inp_shape

        output = predict_with_onnxruntime(onnx_model, random_input, onnx_path=onnx_filename if "cnn" in onnx_filename else None, sess=sess)

        flat_out = output.flatten(flatten_order)  # check order

        for prop_mat, prop_rhs in spec_list:
            vec = prop_mat.dot(flat_out)
            sat = np.all(vec <= prop_rhs)

            if sat:
                print(f"Trial #{trial + 1} found sat case with input {input_list} and output {list(flat_out)}")
                res = 'violated'
                break

        if res == 'violated':
            break

    diff = time.perf_counter() - start
    print(f"Test time: {round(diff, 3)} sec")

    print(f"Result: {res}")

    return res


def main():
    'main entry point'

    trials = 10
    seed = 0
    flatten_order = 'C'

    assert len(sys.argv) >= 4, "expected at least 3 args: <onnx-filename> <vnnlib-filename> <output-filename> " + \
                               f"[<trials>] [<seed>] [<flatten order ('C' or 'F')>] got {len(sys.argv)}"

    onnx_filename = sys.argv[1]
    vnnlib_filename = sys.argv[2]
    output_filename = sys.argv[3]

    if len(sys.argv) > 4:
        trials = int(sys.argv[4])

    if len(sys.argv) > 5:
        seed = int(sys.argv[5])

    if len(sys.argv) > 6:
        flatten_order = sys.argv[6]
        assert flatten_order in ['F', 'C']

    print(f"doing {trials} random trials with random seed {seed}")

    np.random.seed(seed)

    res = run_tests(onnx_filename, vnnlib_filename, trials)

    with open(output_filename, 'w') as f:
        f.write(res)


if __name__ == "__main__":
    main()