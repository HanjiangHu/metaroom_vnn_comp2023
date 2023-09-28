# MetaRoom Benchmark for VNN-COMP 2023
This is the MetaRoom Benchmark for [VNN-COMP 2023](https://sites.google.com/view/vnn2023). This public repo follows the instructions [here](https://github.com/stanleybak/vnncomp2023/issues/2) and [here](https://vnncomp.christopher-brix.de/benchmark/details).

## Introduction
MetaRoom Benchmark is based on [MetaRoom dataset](https://sites.google.com/view/metaroom-dataset/home), which aims at the robustness certification and verification of deep learning based vision models against camera motion perturbation in the indoor robotics application. Specifically, this benchmark focuses on the classification over 20 labels against camera moves within tiny perturbation radii of  translation along z-axis and rotation along y-axis (e.g. 1e-5 m, 2.5e-4 deg). More details about the dataset can be found in [PDF](https://proceedings.mlr.press/v205/hu23b.html). 

## Running without custom projection OP (*for VNN-COMP 2023*)
**In VNN-COMP 2023, the custom projection operator is not involved.**  The input of ONNX is the projected image with the shape of `1*3*32*56` and VNNLIB is the spec over all pixels of the image during the projection within camera perturbation radius of translation along z-axis or rotation along y-axis. The perturbation radius is tiny enough (e.g. 1e-5 m, 2.5e-4 deg),  so it is the ideally equivalent approximation of the one with custom projection OP. 

## Running with custom projection OP
If it is feasible to use the custom projection operator ``ProjectionOp`` in your method, please use ONNX and VNNLIB under `custom_OP` folder. It is recommended for Python-based tools since more demos and tests of the custom projection OP are shown below in detail.

### Details about ONNX and VNNLIB
For the ONNX, the pretrained [neural networks](https://github.com/Verified-Intelligence/auto_LiRPA/blob/master/examples/vision/models/feedforward.py) of `cnn_4layer` and `cnn_6layer` are converted from Pytorch models to ONNX models with customized `torch.nn.module` for image projection operator. The input is the one-dimension relative camera motion around 0 (e.g. [-1e-5, 1e-5]) and the output is for classification over 20 labels. The pretrained models give correct prediction at camera motion movement origins. Run 

``python generate_properties.py XX``

with random seed `XX` to generate random 100 test samples among different models and different perturbation types (y-axis rotation or z-axis translation), shown in `vnnlib/` and `instances.csv`.

### Detailed description of customized ``ProjectionOp`` operator
The customized operator ``ProjectionOp`` is a function with the input of 1-dimension relative camera pose and the output of the projected RGB image given the point cloud, camera pose origin and camera intrinsic parameters. Therefore, we first need to download `dataset.zip` from [here](https://drive.google.com/file/d/1uiuAymh1E4QYAfA_VSblK_iUCxpCB5Dk/view?usp=sharing) and unzip it under the root path. 

To make it easier to implement, we wrap up the projection code into one file ``./randgen/custom_projection.py``. The following packages are required: ``numpy cupy torch torchvision onnxruntime onnxruntime_extensions pickle``.

For the usage of the operator, after loading ``onnx_model``,  just replace `onnxruntime.InferenceSession` with function `custom_ort_session` from ``./randgen/custom_projection.py`` to construct `InferenceSession`. Here is an example in function ``predict_with_onnxruntime`` in ``./randgen/util.py``.
```
import onnxruntime
import onnx
from custom_projection import custom_ort_session

onnx_model = onnx.load(onnx_filename)
# session = onnxruntime.InferenceSession(onnx_model.SerializeToString())
session = custom_ort_session(onnx_path)
```
Run the following `randgen.py` for a demo to generate and pass random trials using the customized `ProjectionOp` operator.

### Test and pass random trials 
The ONNX models can be tested by running 

``python test_onnx.py --onnx_file ./onnx/XXX``

with the camera at the movement origin. Besides, following [randgen](https://github.com/stanleybak/simple_adversarial_generator.git), ONNX and VNNLIB files are also passed random trials. Please note that since the projection operator is customized, it is required to register it again (but only once) in the session when running using `onnxruntime` and `onnxruntime_extensions`. The modified codes are under `randgen` and can be run as 

``python randgen.py ../onnx/XXX ../vnnlib/YYY output_file``



### Citation
If you fine the repo useful, feel free to cite:

H. Hu, C. Liu, and D. Zhao "[Robustness Verification for Perception Models against Camera Motion Perturbations](https://files.sri.inf.ethz.ch/wfvml23/papers/paper_17.pdf)", ICML WFVML 2023
```
@inproceedings{hu2023robustness,
  title={Robustness Verification for Perception Models against Camera Motion Perturbations},
  author={Hu, Hanjiang and Liu, Changliu and Zhao, Ding},
  booktitle={ICML Workshop on Formal Verification of Machine Learning (WFVML)},
  year={2023}
}
```


H. Hu, Z. Liu, L. Li, J. Zhu and D. Zhao "[Robustness Certification of Visual Perception Models via Camera Motion Smoothing](https://proceedings.mlr.press/v205/hu23b.html)", CoRL 2022
```
@InProceedings{pmlr-v205-hu23b,
  title = 	 {Robustness Certification of Visual Perception Models via Camera Motion Smoothing},
  author =       {Hu, Hanjiang and Liu, Zuxin and Li, Linyi and Zhu, Jiacheng and Zhao, Ding},
  booktitle = 	 {Proceedings of The 6th Conference on Robot Learning},
  pages = 	 {1309--1320},
  year = 	 {2023},
  volume = 	 {205},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {14--18 Dec},
  publisher =    {PMLR}
}
```

### Reference
> - [camera-motion-smoothing](https://github.com/HanjiangHu/camera-motion-smoothing)
> - [randgen](https://github.com/stanleybak/simple_adversarial_generator)
> - [vnncomp2022_benchmarks](https://github.com/ChristopherBrix/vnncomp2022_benchmarks)


