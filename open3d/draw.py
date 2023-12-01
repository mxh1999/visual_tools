import open3d as o3d
import numpy as np
from .linemesh import LineMesh

def draw_camera(camera_pose, camera_color=(100, 149, 237), camera_size = 0.5, width=None):
    """
        camera_pose : np.array 4x4 CAMERA to WORLD
    """
    zhui = np.array([[0,0,0], [-camera_size, -camera_size, camera_size*2], [camera_size, -camera_size, camera_size*2], [-camera_size, camera_size, camera_size*2], [camera_size, camera_size, camera_size*2]])
    aa = o3d.geometry.PointCloud(points = o3d.utility.Vector3dVector(zhui))
    aa.transform(camera_pose)
    
    posi = np.array([[camera_pose[0][3],camera_pose[1][3],camera_pose[2][3]]])
    dire = np.array([[camera_pose[0][2],camera_pose[1][2],camera_pose[2][2]]])
    dire = dire + posi
    camera = o3d.geometry.PointCloud()
    camera.points = o3d.utility.Vector3dVector(posi)
    
    color = tuple([x / 255 for x in camera_color])
    if width is not None:
        lines_pcd = LineMesh(points=aa.points, lines=o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3], [0, 4], [1,2], [1,3], [2,4], [3,4]]), colors=color, radius=width)
        return lines_pcd.cylinder_segments
    else:
        lines_pcd = o3d.geometry.LineSet()
        lines_pcd.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3], [0, 4], [1,2], [1,3], [2,4], [3,4]])
        lines_pcd.points = aa.points
        lines_pcd.paint_uniform_color(color)
        return lines_pcd

def draw_boxes_with_thickness(boxes, line_width=0.02):
    """
        boxes : List[open3d.boundingbox]
    """
    results = []
    for box in boxes:
        bbox_lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
        bbox_lines_width = LineMesh(points=bbox_lines.points, lines=bbox_lines.lines, colors=box.color, radius = line_width)
        results += bbox_lines_width.cylinder_segments
    return results

def draw_9dof_boxes(bbox, mode, colors):
    n = bbox.shape[0]
    geo_list = []
    for i in range(n):
        center = bbox[i][:3].reshape(3,1)
        scale = bbox[i][3:6].reshape(3,1)
        rot = bbox[i][6:].reshape(3,1)
        if mode == 'xyz':
            rot_mat = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz(rot)
        elif mode == 'zxy':
            rot_mat = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_zxy(rot)
        else:
            raise NotImplementedError
        geo = o3d.geometry.OrientedBoundingBox(center, rot_mat, scale)
        color = colors[i]
        geo.color = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
        geo_list.append(geo)
    return geo_list

def draw_depth_map(depth_img, extrinsic, depth_instrinc, return_numpy=False):
    """
        depth_img : np.array
        extrinsic : np.array 4x4 CAMERA to WORLD
        depth_intrinsic: np.array 4x4
    """
    depth_point = []
    
    us, vs = np.where(depth_img > 0)
    ds = depth_img[us, vs]
    points = np.stack([vs * ds, us * ds, ds, np.ones((us.shape[0],))], axis=1)
    print(points.shape)
    xyzs = extrinsic @ np.linalg.inv(depth_instrinc) @ points.transpose()
    xyzs = xyzs.transpose()
    depth_point = xyzs[:,:3]

    if return_numpy:
        return depth_point
    depth_point_cloud = o3d.geometry.PointCloud(points = o3d.utility.Vector3dVector(depth_point))
    return depth_point_cloud

