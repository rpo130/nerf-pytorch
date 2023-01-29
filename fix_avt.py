import os
import numpy as np
import json
from scipy.spatial.transform import Rotation as R


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
