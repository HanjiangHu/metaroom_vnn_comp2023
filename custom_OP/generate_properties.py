'''
Hanjiang Hu, for VNN-COMP 2023
May, 2023
'''
import argparse
import numpy as np
import os, shutil
import glob


def write_vnn_spec(dataset, index, eps_factor, dir_path="./vnnlib", prefix="spec", n_class=20):
    onnx_path = dataset[index]
    y = int(onnx_path.split('_')[-1][:-5])
    proj_type = onnx_path.split('_')[1]
    if proj_type == 'tz':
        eps = 0.01 * eps_factor
    elif proj_type == 'ry':
        eps = 0.00436 * eps_factor
    else:
        assert False, f'Unsupported projection type: {proj_type}'
    x_lb = [-eps]
    x_ub = [eps]

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    spec_name = f"{prefix}_idx_{index}_eps_{eps:.8f}.vnnlib"


    spec_path = os.path.join(dir_path, spec_name)

    with open(spec_path, "w") as f:
        f.write(f"; Spec for sample id {index} and epsilon {eps:.8f}\n")

        f.write(f"\n; Definition of input variables\n")
        for i in range(len(x_ub)):
            f.write(f"(declare-const X_{i} Real)\n")

        f.write(f"\n; Definition of output variables\n")
        for i in range(n_class):
            f.write(f"(declare-const Y_{i} Real)\n")

        f.write(f"\n; Definition of input constraints\n")
        for i in range(len(x_ub)):
            f.write(f"(assert (<= X_{i} {x_ub[i]:.8f}))\n")
            f.write(f"(assert (>= X_{i} {x_lb[i]:.8f}))\n")

        f.write(f"\n; Definition of output constraints\n")
        f.write(f"(assert (or\n")
        for i in range(n_class):
            if i == y: continue
            f.write(f"\t(and (>= Y_{i} Y_{y}))\n")
        f.write(f"))\n")
    return spec_name

def main():
    parser = argparse.ArgumentParser(description='VNN spec generator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('seed', type=int, default=42, help='Random seed for idx generation')
    parser.add_argument('--epsilon_factor', type=float, default=0.001, help='The scaled factor epsilon for L_infinity perturbation, tz: 10mm, ry: 0.25 deg')
    parser.add_argument('--n', type=int, default=100, help='The number of specs to generate')
    parser.add_argument("--network_path", type=str, default='./onnx', help="Network to evaluate as .onnx file.")
    parser.add_argument("--spec_path", type=str, default='./vnnlib', help="Network to evaluate as .vnnlib file.")
    parser.add_argument("--instances", type=str, default="./instances.csv", help="Path to instances file")
    parser.add_argument('--time_out', type=float, default=210.0, help='the mean used to normalize the data with')

    args = parser.parse_args()

    shutil.rmtree(args.spec_path)

    nets = sorted(glob.glob(args.network_path + '/*'))
    np.random.seed(args.seed)
    idxs = list(np.random.choice(len(nets), args.n, replace=False))

    i = 0
    with open(args.instances, "w") as f:
        while i<len(idxs):
            idx = idxs[i]
            i += 1
            spec_name = write_vnn_spec(nets, idx, args.epsilon_factor, dir_path=args.spec_path, n_class=20)
            f.write(f"{nets[idx]},{os.path.join(args.spec_path,spec_name)},{args.time_out:.1f}\n")
if __name__ == "__main__":
    main()