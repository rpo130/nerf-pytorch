import os
import numpy as np
import json
from scipy.spatial.transform import Rotation as R

def fix_pose():
    basedir = 'data/avt_data_glass_20230118_1'
    with open(os.path.join(basedir, 'transforms.json'), 'r') as fp:
        meta = json.load(fp)
        
        for frame in meta['frames'][::1]:
            T_cam_to_world = np.array(frame['transform_matrix'])
            T_cam_face_to_world = T_cam_to_world

            T_img_to_cam_face = np.eye(4) 
            T_img_to_cam_face[:3, :3] = R.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()
            T_cam_to_world = T_cam_face_to_world @ T_img_to_cam_face
            frame['transform_matrix'] = T_cam_to_world.tolist()

        json_object = json.dumps(meta, indent=2)
        with open(os.path.join(basedir, "transforms_fix.json"), "w") as outfile:
            outfile.write(json_object)

def fix_boarder():
    import imageio.v2 as imageio
    basedir = 'data/avt_data_glass_20230118_2/'
    
    img_file_path = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    
    
    for fp in img_file_path:
        print(fp)
        img = imageio.imread(fp)
        # crop_image = img[90:390, 120:520, ...]
        print(img)
        mask_image = np.zeros_like(img)
        mask_image[90:390, 120:520, ...] = 1
        mask_image = img * mask_image
        import matplotlib.pyplot as plt
        plt.imshow(mask_image)
        plt.show()

def crop_img(basedir, factor):
    import imageio.v2 as imageio

    newbasedir = basedir + '_crop' + str(factor[0]) + "-" + str(factor[1])

    if os.path.exists(newbasedir):
        print('exist')
        return
    
    os.makedirs(newbasedir)

    # images
    img_file_name = [f for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    
    for fn in img_file_name:
        fp = os.path.join(basedir, 'images', fn)
        img = imageio.imread(fp)
        o_shape = img.shape
        o_H,o_W = o_shape[0], o_shape[1]
        crop_height_boarder = o_H * (factor[1] - factor[0]) / factor[1] / 2
        crop_width_boarder = o_W * (factor[1] - factor[0]) / factor[1] / 2
        crop_height_boarder = int(crop_height_boarder)
        crop_width_boarder = int(crop_width_boarder)
        img = img[crop_height_boarder:-crop_height_boarder, crop_width_boarder:-crop_width_boarder, ...]

        nfp = os.path.join(newbasedir, 'images', fn)
        os.makedirs(os.path.join(newbasedir, 'images'), exist_ok=True)
        imageio.imwrite(nfp, img)
        print(f'write img {nfp}, shape {o_shape} -> {img.shape}')

    # np
    np_file_name = [f for f in sorted(os.listdir(os.path.join(basedir, 'np'))) \
            if f.endswith('npy')]
    for fn in np_file_name:
        fp = os.path.join(basedir, 'np', fn)
        n = np.load(fp)
        o_shape = n.shape
        o_H,o_W = o_shape[0], o_shape[1]
        crop_height_boarder = o_H * (factor[1] - factor[0]) / factor[1] / 2
        crop_width_boarder = o_W * (factor[1] - factor[0]) / factor[1] / 2
        crop_height_boarder = int(crop_height_boarder)
        crop_width_boarder = int(crop_width_boarder)
        n = n[crop_height_boarder:-crop_height_boarder, crop_width_boarder:-crop_width_boarder, ...]

        nfp = os.path.join(newbasedir, 'np', fn)
        os.makedirs(os.path.join(newbasedir, 'np'), exist_ok=True)
        np.save(nfp, n)
        print(f'write npy {nfp}, shape {o_shape} -> {n.shape}')


    # json
    json_file_name = [f for f in sorted(os.listdir(os.path.join(basedir))) \
            if f.endswith('json')]
    for fn in json_file_name:
        fp = os.path.join(basedir, fn)
        with open(fp, 'r') as f:
            tran = json.load(f)
        tran['fx'] = tran['fx']
        tran['fy'] = tran['fy']
        tran['cx'] = tran['cx'] * factor[0] / factor[1]
        tran['cy'] = tran['cy'] * factor[0] / factor[1]


        nfp = os.path.join(newbasedir, fn)
        with open(nfp, 'w') as f:
            f.write(json.dumps(tran, indent=2))
            print(f'write json {nfp}')
    
    print('finish')

def center(basedir, T_scene_to_world):
    with open(os.path.join(basedir, 'transforms.json'), 'r') as fp:
        meta = json.load(fp)
        
        for frame in meta['frames'][::1]:
            T_cam_to_world = np.array(frame['transform_matrix'])
            T_cam_face_to_world = T_cam_to_world

            T_cam_face_to_scene = np.linalg.inv(T_scene_to_world) @ T_cam_face_to_world
            
            T_cam_to_world = T_cam_face_to_scene
            frame['transform_matrix'] = T_cam_to_world.tolist()

        json_object = json.dumps(meta, indent=2)
        with open(os.path.join(basedir, "transforms_center.json"), "w") as outfile:
            outfile.write(json_object)

def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0) # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0)) # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0) # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z)) # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x) # (3)

    pose_avg = np.stack([x, y, z, center], 1) # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg # convert to homogeneous coordinate for faster computation
                                 # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    return poses_centered, pose_avg

def cross_point(point1, direction1, point2, direction2):
    direction1 = normalize(direction1)
    direction2 = normalize(direction2)

    for i in range(100000):
        next_point = point1 + direction1 * 0.001 * i
        cos = np.dot(normalize((next_point - point2)), direction2)
        if (abs(cos) - 1) < 1e-8:
            return next_point

        # next_point = point1 + direction1 * -0.0001 * i
        # cos = np.dot(normalize((next_point - point2)), direction2)
        # if (abs(cos) - 1) < 1e-8:
        #     return next_point
    print('not found')
    return None


# crop_img('data/avt_kinect_glass_20230131_1', (8,10))
# crop_img('data/avt_data_glass_20230118_1', (5,8))
# crop_img('data/avt_data_glass_light_20230115_1', (5,8))

# T_s2w = np.eye(4)
# T_s2w[0:3,3] = [0.6, 0, 0]
# center('data/avt_20230218_glass_6', T_s2w)

if __name__ == "__main__":
    basedir = 'data/avt_20230218_glass_6'
    with open(os.path.join(basedir, 'transforms.json'), 'r') as fp:
        meta = json.load(fp)
        poses = []
        for frame in meta['frames'][::1]:
            T_cam_to_world = np.array(frame['transform_matrix'])
            poses.append(T_cam_to_world)

    poses = np.array(poses)

    poses_ori = poses.copy()

    #rdf -> rub, r:right d:down u:up f:forward, point out
    poses = np.concatenate([poses[..., 0:1], -poses[..., 1:3], poses[..., 3:4]], -1)
    pose_avt = np.eye(4)
    poses[..., :3, :4], pose_avt[:3, :4] = center_poses(poses[..., :3, :4])

    from display import *
    print(cross_point(poses_ori[0][:3, 3], poses_ori[0][:3, 2], pose_avt[:3, 3], pose_avt[:3, 2]))
    fig = plt.figure()
    draw_pose_list(fig, poses[::2])

    plt.show()