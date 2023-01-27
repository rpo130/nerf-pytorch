import os
import torch
import numpy as np
import imageio 
import json
import cv2
from load_blender import trans_t, rot_phi, rot_theta, pose_spherical
import random
from scipy.spatial.transform import Rotation as R

def gen_split(imgs, testskip):
    i_train = []
    i_val = [] 
    i_test = []

    for i in range(len(imgs)):
        b = random.randint(0,9)
        if b >=0 and b<8:
            i_train.append(i)
        elif b>=8 and b<9:
            i_val.append(i)
        else:
            i_test.append(i)

    if len(i_train) == 0:
        i_train.append(random.randint(0, len(imgs)-1))  
    if len(i_val) == 0:
        i_val.append(random.randint(0, len(imgs)-1))
    if len(i_test) == 0:
        i_test.append(random.randint(0, len(imgs)-1))

    if testskip == 0:
        skip = 1
    else:
        skip = testskip

    i_val = i_val[::skip]
    i_test = i_test[::skip]

    i_train = sorted(i_train)
    i_val = sorted(i_val)
    i_test = sorted(i_test)
    return [i_train, i_val, i_test]

def load_avt_data(basedir, half_res=False, testskip=1):
    with open(os.path.join(basedir, 'transforms.json'), 'r') as fp:
        meta = json.load(fp)

    all_imgs = []
    all_poses = []
    imgs = []
    poses = []
        
    for frame in meta['frames'][::1]:
        file_path = frame['file_path']
        if '.png' not in file_path:
            file_path = file_path + '.png'
        fname = os.path.join(basedir, file_path)
        imgs.append(imageio.imread(fname))
        T_cam_face_to_world = np.array(frame['transform_matrix'])
        #fix poses
        T_img_to_cam_face = np.eye(4) 
        T_img_to_cam_face[:3, :3] = R.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()
        T_cam_to_world = T_cam_face_to_world @ T_img_to_cam_face
        poses.append(T_cam_to_world)
    imgs = (np.array(imgs) / 255.).astype(np.float32)
    poses = np.array(poses).astype(np.float32)
    all_imgs.append(imgs)
    all_poses.append(poses)
        
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    # camera_angle_x = float(meta['camera_angle_x'])
    # focal = .5 * W / np.tan(.5 * camera_angle_x)
    focal = float(meta['fx'])
    
    imgs = imgs[...,:3]

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        
    fx = meta['fx']
    fy = meta['fy']
    cx = meta['cx']
    cy = meta['cy']

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    # K = None
        

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    i_split = gen_split(imgs, testskip)

    return imgs, poses, render_poses, [H, W, focal], i_split, K