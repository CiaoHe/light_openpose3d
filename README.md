# Real-time 3D Multi-person Pose Estimation Demo

This repository contains 3D multi-person pose estimation demo in PyTorch. Intel OpenVINO&trade; backend can be used for fast inference on CPU. This demo is based on [Lightweight OpenPose](https://arxiv.org/pdf/1811.12004.pdf) and [Single-Shot Multi-Person 3D Pose Estimation From Monocular RGB](https://arxiv.org/pdf/1712.03453.pdf) papers. It detects 2D coordinates of up to 18 types of keypoints: ears, eyes, nose, neck, shoulders, elbows, wrists, hips, knees, and ankles, as well as their 3D coordinates. It was trained on [MS COCO](http://cocodataset.org/#home) and [CMU Panoptic](http://domedb.perception.cs.cmu.edu/) datasets and achieves 100 mm MPJPE (mean per joint position error) on CMU Panoptic subset. *This repository significantly overlaps with https://github.com/opencv/open_model_zoo/, however contains just the necessary code for 3D human pose estimation demo.*

<p align="center">
  <img src="data/human_pose_estimation_3d_demo.jpg" />
</p>

> The major part of this work was done by [Mariia Ageeva](https://github.com/marrmar), when she was the :top::rocket::fire: intern at Intel.

## Table of Contents

* [Requirements](#requirements)
* [Prerequisites](#prerequisites)
* [Pre-trained model](#pre-trained-model)
* [Running](#running)
* [Inference with OpenVINO](#inference-openvino)
* [Inference with TensorRT](#inference-tensorrt)

## Requirements
* Python 3.5 (or above)
* CMake 3.10 (or above)
* C++ Compiler (g++ or MSVC)
* OpenCV 4.0 (or above)

> [Optional] [Intel OpenVINO](https://software.intel.com/en-us/openvino-toolkit) for fast inference on CPU.
> [Optional] [NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) for fast inference on Jetson.

## Prerequisites
1. Install requirements:
```
pip install -r requirements.txt
```
2. Build `pose_extractor` module:
```
python setup.py build_ext
```
3. Add build folder to `PYTHONPATH`:
```
export PYTHONPATH=pose_extractor/build/:$PYTHONPATH
```

## Pre-trained model <a name="pre-trained-model"/>
Already downloaded at `human-pose-estimation-3d.pth`.
~~Pre-trained model is available at [Google Drive](https://drive.google.com/file/d/1niBUbUecPhKt3GyeDNukobL4OQ3jqssH/view?usp=sharing).~~

## Running

### Demo Running
To run the demo_img script, just simply run:
```
python demo_img.py
```

Detailed about main function `inference_single_image()`
```python
def inference_single_image(
    images:Union[str, List[str], np.ndarray],
    model:str, 
    device:str='GPU', 
    height_size:int=256, 
    extrinsics_path:str=None, 
    fx:np.float32=-1, 
    use_tensorrt:bool=False,
    stride:int=8,
    out_dir:str=None,
):  
    """
    Args:
        images (Union[str, List[str]]): input image(s)
        model (str): Required. Path to checkpoint with a trained model
        device (str, optional): Defaults to 'GPU'.
        height_size (int, optional): Network input layer height size.. Defaults to 256.
        extrinsics_path (str, optional): Path to file with camera extrinsics. Defaults to None.
        fx (np.float32, optional): Camera focal length. Defaults to -1.
        use_tensorrt (bool, optional): Optional. Run network with OpenVINO as inference engine. Defaults to False.
        stride (int, optional): Defaults to 8.
        out_dir (str, optional): Optional. Path to output directory. Defaults to None.
    """
```

### Original Demo Running [skip this]
To run the demo, pass path to the pre-trained checkpoint and camera id (or path to video file):
```
python demo.py --model human-pose-estimation-3d.pth --video 0
```
> Camera can capture scene under different view angles, so for correct scene visualization, please pass camera extrinsics and focal length with `--extrinsics` and `--fx` options correspondingly (extrinsics sample format can be found in data folder). In case no camera parameters provided, demo will use the default ones.
