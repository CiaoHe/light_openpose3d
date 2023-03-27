from typing import Union, List
import json
import os

import cv2
import numpy as np

from modules.input_reader import ImageReader
from modules.draw import Plotter3d, draw_poses
from modules.parse_poses import parse_poses


def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

    return poses_3d

def build_model(model:str, device:str, use_tensorrt:bool=False):
    from modules.inference_engine_pytorch import InferenceEnginePyTorch
    net = InferenceEnginePyTorch(model, device, use_tensorrt=use_tensorrt)
    return net


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
        stride (int, optional): _description_. Defaults to 8.
    """
    net = build_model(model, device, use_tensorrt)

    if isinstance(images, str):
        if images == '':
            raise ValueError('--image has to be provided')
        frame_provider = ImageReader([images])
    elif isinstance(images, List):
        if len(images) == 0:
            raise ValueError('--image has to be provided')
        frame_provider = ImageReader(images)
    elif isinstance(images, np.ndarray):
        if len(images.shape) == 3:
            frame_provider = [images]
        elif len(images.shape) == 4:
            frame_provider = images
    else:
        raise ValueError('Not supported image type: {}'.format(type(images)))
        
    
    base_height = height_size
    canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
    plotter = Plotter3d(canvas_3d.shape[:2])

    # prepare camera intrinsics
    if extrinsics_path is None:
        extrinsics_path = os.path.join('data', 'extrinsics.json')
    with open(extrinsics_path, 'r') as f:
        extrinsics = json.load(f)
    # R = np.array(extrinsics['R'], dtype=np.float32)
    # t = np.array(extrinsics['t'], dtype=np.float32)
    # !! For Debug: fix R,t
    R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    t = np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape((3, 1))

    # inference
    for frame in frame_provider:
        if frame is None:
            break
        # In case Focal length is unknown
        if fx < 0:  
            fx = np.float32(0.8 * frame.shape[1])
        
        input_scale = base_height / frame.shape[0]
        scaled_img = cv2.resize(frame, dsize=None, fx=input_scale, fy=input_scale)
        scaled_img = scaled_img[:, 0:scaled_img.shape[1] - (scaled_img.shape[1] % stride)]  # better to pad, but cut out for demo

        inference_result = net.infer(scaled_img)
        poses_3d, poses_2d = parse_poses(inference_result, input_scale, stride, fx, is_video=False)
        edges = []
        if len(poses_3d):
            poses_3d = rotate_poses(poses_3d, R, t)
            poses_3d_copy = poses_3d.copy()
            x = poses_3d_copy[:, 0::4]
            y = poses_3d_copy[:, 1::4]
            z = poses_3d_copy[:, 2::4]
            poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = -z, x, -y

            poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
            edges = (Plotter3d.SKELETON_EDGES + 19 * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
            # poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
            # # drop the 2nd joint
            # poses_3d = poses_3d[:, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], :]
            # edges = (Plotter3d.SKELETON_EDGES + 18 * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
            
        if out_dir is not None:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            # save 3d canvas
            plotter.plot(canvas_3d, poses_3d, edges)
            canvas_3d_file = os.path.join(out_dir, 'canvas_3d.png')
            cv2.imwrite(canvas_3d_file, canvas_3d)
            
            # save 2d canvas
            draw_poses(frame, poses_2d)
            canvas_2d_file = os.path.join(out_dir, 'canvas_2d.png')
            cv2.imwrite(canvas_2d_file, frame)

if __name__ == '__main__':
    inference_single_image(
        images='human3d.png',
        model='human-pose-estimation-3d.pth',
        out_dir='vis_result',
    )