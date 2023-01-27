import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

def gen_camera_point(T_):
    size = 0.1
    X, Y = np.meshgrid(np.linspace(-size, size, 3), np.linspace(-size, size, 3))
    Z = np.zeros_like(X)
    Z[1, 1] = size
    b = np.stack([X, Y, Z, np.ones_like(X)], -1)
    b = b @ T_.T
    return b

def draw_camera_shape(ax, b):
    ax.plot_wireframe(b[..., 0], b[..., 1], b[..., 2], colors=['red', 'green', 'blue', 
                                                               'black', 'black', 'black'])

def draw_camera_coor(ax, ori, x,y,z):
    ax.quiver3D(ori[0],ori[1],ori[2],x[0],x[1],x[2], length=0.1, colors='red')
    ax.quiver3D(ori[0],ori[1],ori[2],y[0],y[1],y[2], length=0.1, colors='green')
    ax.quiver3D(ori[0],ori[1],ori[2],z[0],z[1],z[2], length=0.1, colors='blue')
    
def draw_pose_list(fig, pose_list):
    ax = fig.add_subplot(projection='3d')

    ori = np.array([0.,0.,0.,1.])

    ax.scatter(ori[0], ori[1], ori[1], marker='o')

    for pose in pose_list:
        a = pose @ ori
        ax.scatter(a[0], a[1], a[2])
        draw_camera_shape(ax, gen_camera_point(pose))
        draw_camera_coor(ax, pose@ori, pose@np.array([1.,0.,0.,0.]), pose@np.array([0.,1.,0.,0.]), pose@np.array([0.,0.,1.,0.]))

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


if __name__ == "__main__":
    with open('./data/avt_data_glass_20230118_1/transforms.json', 'r') as f:
        j = f.read()
    import json
    j = json.loads(j)
    print(j)
    pose_list = []
    for i in j['frames']:
        pose_list.append(np.array(i['transform_matrix']))
    fig = plt.figure()
    draw_pose_list(fig, pose_list)

    plt.show()