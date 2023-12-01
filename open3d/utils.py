import open3d as o3d
import os
import numpy as np
from .linemesh import LineMesh

def print_to_file(obj_list, filepath, size=(1296, 1296), work_dir = './', force_save=False):
    height, width = size
    vis = o3d.visualization.Visualizer()
    vis.create_window(width = width, height = height, visible=True) #works for me with False, on some systems needs to be true
    render = vis.get_render_option()
    if os.path.exists(os.path.join(work_dir, 'render.json')) and not force_save:
        render.load_from_json(os.path.join(work_dir, 'render.json'))
    else:
        render.save_to_json(os.path.join(work_dir, 'render.json'))
    render.background_color = np.asarray([255, 255, 255])
    vis.update_renderer()
    # render.light_on = True
    # print(vis.get_render_option().background_color)
    for obj in obj_list:
        vis.add_geometry(obj)
        vis.update_geometry(obj)
    vis.poll_events()
    vis.update_renderer()
    ctr = vis.get_view_control()
    if os.path.exists(os.path.join(work_dir, 'camera.json')) and not force_save:
        cam = o3d.io.read_pinhole_camera_parameters(os.path.join(work_dir, 'camera.json'))
        ctr.convert_from_pinhole_camera_parameters(cam)
    else:
        cam = ctr.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(os.path.join(work_dir, 'camera.json'), cam)
    vis.capture_screen_image(filepath, do_render=True)
    vis.destroy_window()
    vis.close()
