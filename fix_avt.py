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

