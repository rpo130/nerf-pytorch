import os
import torch
import numpy as np
import imageio 
import json
import cv2
from load_blender import trans_t, rot_phi, rot_theta, pose_spherical
import random
from scipy.spatial.transform import Rotation as R

with open(os.path.join('data/avt_data_glass_20230118_1', 'transforms.json'), 'r') as fp:
    meta = json.load(fp)
    
    for frame in meta['frames'][::1]:
        T_cam_to_world = np.array(frame['transform_matrix'])
        T_cam_face_to_world = T_cam_to_world

        T_img_to_cam_face = np.eye(4) 
        T_img_to_cam_face[:3, :3] = R.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()
        T_cam_to_world = T_cam_face_to_world @ T_img_to_cam_face
        frame['transform_matrix'] = T_cam_to_world.tolist()

    json_object = json.dumps(meta, indent=2)
    with open(os.path.join('data/avt_data_glass_20230118_1', "transforms_fix.json"), "w") as outfile:
        outfile.write(json_object)
