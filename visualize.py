import open3d
import json
import os
import argparse
import numpy as np

idx = 0
label_files = []
pcd_files = []
to_read = False

def text_3d(text, pos, direction=[1,0,0], degree=0.0, density=10 ,font=os.path.dirname(__file__) +'/DejaVuSansMono.ttf', font_size=50):
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: open3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    from PIL import Image, ImageFont, ImageDraw
    from pyquaternion import Quaternion

    font_obj = ImageFont.truetype(font = font, size = font_size * density)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = open3d.geometry.PointCloud()
    pcd.colors = open3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = open3d.utility.Vector3dVector(indices / 100.0 / density)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    pcd.paint_uniform_color([1, 1, 1])
    return pcd

def read_json_label(path):
    # *读取单个注释文件
    with open(path) as fin:
        label = json.load(fin)
    
    if 'objs' in label:
        label = label['objs']
    
    bboxes = []

    for obj in label:
        bbox = [0] * 9
        bbox[0] = obj['psr']['position']['x']
        bbox[1] = obj['psr']['position']['y']
        bbox[2] = obj['psr']['position']['z']

        bbox[3] = obj['psr']['scale']['x']
        bbox[4] = obj['psr']['scale']['y']
        bbox[5] = obj['psr']['scale']['z']

        bbox[6] = obj['psr']['rotation']['z']

        bbox[7] = obj['obj_id']

        bbox[8] = obj['obj_type']

        bboxes.append(bbox)

    return bboxes


def read_display_pcd_pc(path, vis):
    pcd=open3d.io.read_point_cloud(path)       
    pcd.paint_uniform_color([1, 1, 1])
    vis.add_geometry(pcd)

def read_display_bbox(path, vis):
    bboxes = read_json_label(path)

    for i, bbox in enumerate(bboxes):
        b = open3d.geometry.OrientedBoundingBox()
        b.center = bbox[:3]
        b.extent = bbox[3:6]
        b.color = [1, 0, 0]
        R = open3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz((0, 0, bbox[6]))
        b.rotate(R, b.center)
        label = text_3d(bbox[7] + '_' + bbox[8], b.center + [0,-1.5,2])
        label.paint_uniform_color([1, 0, 0])
        vis.add_geometry(label)
        vis.add_geometry(b)
    

def viewpoint(vis):
    param = open3d.io.read_pinhole_camera_parameters(os.path.dirname(__file__) +'/viewpoint.json')
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(param)
    
   
def key_play_forward_callback(vis):    
    global idx, to_read
    to_read = True
    vis.clear_geometries()

    mesh = open3d.geometry.TriangleMesh.create_coordinate_frame()
    vis.add_geometry(mesh)

    read_display_pcd_pc(pcd_files[idx], vis)
    read_display_bbox(label_files[idx], vis)
    idx += 1
    if idx >= len(pcd_files):
        idx = len(pcd_files) - 1
    
    viewpoint(vis)
    
    vis.poll_events()
    vis.update_renderer()

    return True

def key_play_back_callback(vis):
    global idx
    vis.clear_geometries()


    read_display_pcd_pc(pcd_files[idx], vis)
    read_display_bbox(label_files[idx], vis)
    idx -= 1
    if idx < 0:
        idx = 0

    viewpoint(vis)

    vis.poll_events()
    vis.update_renderer()

    return True

def key_save_callback(vis):
    global to_read 
    if to_read:
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        open3d.io.write_pinhole_camera_parameters(os.path.dirname(__file__) +'/viewpoint.json', param)
    return True

def key_forward_callback(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(1, 0, 0)
    return True

def key_back_callback(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(-1, 0, 0)
    return True

def key_left_callback(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(0, -1, 0)
    return True

def key_right_callback(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(0, 1, 0)
    return True

def key_up_callback(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(0, 0, 1)
    return True

def key_down_callback(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(0, 0, -1)
    return True

def key_look_up_callback(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_rotate(0,-10)
    return True

def key_look_down_callback(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_rotate(0,10)
    return True

def key_look_right_callback(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_rotate(20,0)
    return True

def key_look_left_callback(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_rotate(-20,0)
    # ctr.set_up([0,0,1])
    return True

def key_reset_up_callback(vis):
    ctr = vis.get_view_control()
    ctr.set_up([0,0,1])
    ctr.reset_camera_local_rotate()
    return True

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='pc visualize')
    parser.add_argument('scene_dir', type=str, default='./scene/', help='scene dir')
    args = parser.parse_args()

    scene_dir = args.scene_dir
    

    for file in os.listdir(scene_dir + 'label/'):
        file_name = os.path.splitext(file)[0]
        label_files.append(scene_dir + 'label/' + file_name + '.json')
        pcd_files.append(scene_dir + 'lidar/' + file_name + '.pcd')


    vis=open3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="point_cloud", width=1920, height=1080)
    
    opt=vis.get_render_option()  
    opt.load_from_json(os.path.dirname(__file__) + './RenderOption.json')

    vis.reset_view_point(True)
    vis.register_key_callback(ord('E'), key_play_forward_callback)
    vis.register_key_callback(ord('Q'), key_play_back_callback)
    vis.register_key_callback(ord('W'), key_forward_callback)
    vis.register_key_callback(ord('S'), key_back_callback)
    vis.register_key_callback(ord('A'), key_left_callback)
    vis.register_key_callback(ord('D'), key_right_callback)
    vis.register_key_callback(ord(' '), key_up_callback)
    vis.register_key_callback(ord('C'), key_down_callback)
    vis.register_key_callback(ord('I'), key_look_up_callback)
    vis.register_key_callback(ord('K'), key_look_down_callback)
    vis.register_key_callback(ord('J'), key_look_left_callback)
    vis.register_key_callback(ord('L'), key_look_right_callback)
    vis.register_key_callback(ord('F'), key_reset_up_callback)





    vis.register_animation_callback(key_save_callback)
    
    

    vis.run()