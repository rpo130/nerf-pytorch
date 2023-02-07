import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import pytransform3d.transformations as pt
import pytransform3d.camera as pc
import pytransform3d.plot_utils as pp


def draw_camera_shape(ax, pose):
    scale = 0.2
    pt.plot_transform(ax=ax, A2B=pose, s=scale)
    # sensor_size = np.array([0.036, 0.024])
    # intrinsic_matrix = np.array([
    #     [0.05, 0, sensor_size[0] / 2.0],
    #     [0, 0.05, sensor_size[1] / 2.0],
    #     [0, 0, 1]
    # ])
    # virtual_image_distance = 1
    # pc.plot_camera(ax, 
    #                 M=intrinsic_matrix, 
    #                 cam2world=pose,
    #                 virtual_image_distance=virtual_image_distance,
    #                 sensor_size=sensor_size)
    
def draw_pose_list(fig, pose_list):
    ax = pp.make_3d_axis(ax_s=1)
    ax.set_xlim(0,1)
    ax.set_ylim(-0.5,0.5)
    ax.set_zlim(0,1)
    for pose in pose_list:
        draw_camera_shape(ax, pose)

    plt.show()

def show_pose_entry():
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

def draw(fig, axs, data, expand, iter, test_i):
    ori = f'./data/{data}/np/{test_i*8}_Depth.npy'
    ori_color = f'./data/{data}/np/{test_i*8}_Color.npy'
    gen = f'./logs/{expand}/testset_{iter:06d}/{test_i:03d}_depth.npy'
    opt = f'./logs/{expand}/testset_{iter:06d}/{test_i:03d}_depth_ff.npy'

    ax = axs[0,0]
    img_ori = np.load(ori)
    img = img_ori
    im = ax.imshow(img, cmap='gray')
    ax.set_title('ori depth')

    ax = axs[0,1]
    img_gen = np.load(gen)
    img = img_gen
    im = ax.imshow(img, cmap='gray')
    ax.set_title('vanilla')

    ax = axs[0,2]
    img_dex = np.load(opt)
    img = img_dex
    im = ax.imshow(img, cmap='gray')
    ax.set_title('dex')

    ax = axs[1,0]
    img_ori_color = np.load(ori_color)
    img = img_ori_color
    im = ax.imshow(img)
    ax.set_title('ori color')

    ax = axs[1,1]
    img = img_ori - img_gen
    img = np.abs(img)
    im = ax.imshow(img, cmap='gray')
    ax.set_title('diff ori vanilla')

    ax = axs[1,2]
    img = img_ori - img_dex
    img = np.abs(img)
    im = ax.imshow(img, cmap='gray')
    ax.set_title('diff ori dex')

    fig.canvas.draw_idle()


def compare_depth():
    d = {
        'avt_data_glass_20230204_1' : '1280x720 normal light',
        'avt_data_glass_20230204_2' : '1280x720 dark light',
        'avt_data_glass_20230204_3' : '1280x720 high light',
        'avt_data_glass_20230204_4' : '640x480 high light',
        'avt_data_glass_20230204_5' : '640x480 high light, background',
        'avt_data_glass_20230204_6' : '1280x720 high light, background',
        'avt_data_glass_20230204_7' : '640x480 high light, background, warm light',
        'avt_data_glass_20230204_8' : '1280x720 high light, background, warm light',
        }

    data_name = 'avt_data_glass_20230204_7'
    expand_name = 'avt_data_glass_20230204_7'
    iter = 200000
    test_i = 0

    fig = plt.figure()
    fig.subplots_adjust(bottom=0.2)
    fig.suptitle(f'{d[data_name]}')
    axs = fig.subplots(2,3)

    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    cmap=cm.get_cmap('gray')
    normalizer=Normalize(0,1)
    im=cm.ScalarMappable(norm=normalizer, cmap=cmap)
    fig.colorbar(im, ax=axs.ravel().tolist())

    draw(fig, axs, data_name, expand_name,iter, test_i)

    from matplotlib.widgets import Slider
    ax_slider = plt.axes([0.20, 0.01, 0.65, 0.03])
    slider = Slider(ax_slider, 'test_image', 0,10, valinit=0, valstep=1)
    
    def update(val):
        nonlocal test_i
        test_i = int(val)
        draw(fig, axs, data_name, expand_name,iter, test_i)
    slider.on_changed(update)

    plt.show()

if __name__ == "__main__":
    # show_pose_entry()
    compare_depth()