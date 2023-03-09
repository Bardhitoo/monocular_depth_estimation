# Monocular Depth Estimation

## Introduction

Autonomous driving technology is advancing at a rapid pace and has the potential to revolutionize the transportation industry. One of the crucial components of autonomous driving is the ability to accurately estimate the depth of objects in the environment. In this project, I propose to develop a depth-estimation system using neural networks in C++, with the ultimate goal of integrating it into an autonomous driving system.

## Methodology
I will be utilizing state-of-the-art monocular depth estimation models such as monodepthv2 which leverages information from both the RGB and semantic segmentation to approximate depth from 2D images. The implementation of the depth estimation model is done in C++ to ensure high performance and real-time processing.

Model building and training is done in Python, following the state of current existing research in the field. This repository is based and builds on [MonoDepth2](https://github.com/nianticlabs/monodepth2). 

As C++/OpenCV doesn't directly enable running inference of the pytorch/ tf model, there are some intermediary steps that enable us to convert models to something that is runnable by OpenCV. This intermediary file is known as the `*.onnx` file, which essentially contains information about the network, and training weights, and is able to make inferences in C++/OpenCV utilizing the `dnn` module. The result can be seen in the image down below:

<img src="./figs/short_depthEstimation.gif" width="800"> 
<img src="./figs/short_carsRunning.gif" width="800"> 

The next step in this process was to enable the model to run on GPU by setting up the OpenCV-DNN module with CUDA backend support in Windows. This optimization improved the inference speed significantly. 

My ultimate goal is to retrain a model from the NYU-v2 and KITTI datasets. These datasets contain thousands of annotated images with corresponding depth maps, and they are commonly used for monocular depth estimation tasks. To achieve this, I plan to use transfer learning to fine-tune the pre-trained models provided in the "monodepth2" repository.