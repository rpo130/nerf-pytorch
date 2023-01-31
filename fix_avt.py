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

crop_img('data/avt_kinect_glass_20230131_1', (8,10))
crop_img('data/avt_data_glass_20230118_1', (5,8))
