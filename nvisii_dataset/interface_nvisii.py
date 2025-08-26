import os
import sys
if sys.platform == 'win32':
    from pyglet.gl.wgl import *
else:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import cv2
import open3d as o3d

import glob
import yaml
import gzip
import argparse

import trimesh


class NvisiiScene():
    def __init__(self, scene_root_path, static_equipment=True, labeled=True, check_consistency=True, real_assemblies=False):
        self.root = scene_root_path
        self.static_equipment = static_equipment
        self.labeled = labeled
        self.real_assemblies = real_assemblies

        self.camera_poses = self.parse_camera_poses()
        self.camera_intrinsics = self.parse_camera_intrinsics()
        self.img_paths = self.find_img_paths()
        self.monodepth_paths = self.find_monodepth_paths()
        self.equipment_ids = self.parse_equipment_ids()
        self.equipment_poses = self.parse_equipment_poses()

        if self.labeled:
            self.segmentation_paths = self.find_segmentation_paths()
            self.equipment_point_paths = self.find_equipment_point_paths()
            self.depth_paths, self.depth_bounds_paths = self.find_depth_paths()

        # If we have fewer monodepth images, remove the additional items from all other lists
        # if len(self.monodepth_paths) < len(self.img_paths):
        #     img_ids = list(self.img_paths.keys())
        #     for img_id in img_ids:
        #         if img_id not in self.monodepth_paths.keys():
        #             del self.camera_poses[img_id]
        #             del self.camera_intrinsics[img_id]
        #             del self.img_paths[img_id]
        #             if self.labeled:
        #                 del self.segmentation_paths[img_id]
        #                 del self.equipment_point_paths[img_id]
        #                 del self.depth_paths[img_id]
        #                 del self.depth_bounds_paths[img_id]


        if self.real_assemblies:
            self.assembly_paths = self.find_assembly_paths()

        # Ensure all camera-related files are the same length
        if check_consistency:
            if self.labeled:
                assert all(keys == self.camera_poses.keys() for keys in [self.camera_intrinsics.keys(), self.img_paths.keys(), self.segmentation_paths.keys(), self.equipment_point_paths.keys()]), 'Camera-related files are not the same length'
            else:
                assert all(keys == self.camera_poses.keys() for keys in [self.camera_intrinsics.keys(), self.img_paths.keys()]), 'Camera-related files are not the same length'

        # Cache images
        self.imgs = {}
        self.monodepths = {}
        self.segmentations = {}
        self.equipment_points = {}
        self.depths = {}
        self.assembly_predictions = {}

        # Optionally, part names may be provided
        self.part_names = []
 
    def parse_equipment_ids(self):
        with open(os.path.join(self.root, 'entity_ids.yaml')) as f:
            equipment_ids = yaml.safe_load(f)
        return equipment_ids

    def parse_equipment_poses(self):
        equipment_poses = {}
        for equipment_name in self.get_present_equipment_names() + ['root']:
            equipment_pose_paths = glob.glob(os.path.join(self.root, 'equipment_pose', equipment_name, '*.npy'))
            equipment_pose_paths.sort()
            equipment_poses[equipment_name] = {}
            for equipment_pose_path in equipment_pose_paths:
                equipment_pose = np.load(equipment_pose_path)
                equipment_pose_id = int(os.path.basename(equipment_pose_path).split('.')[0])
                dict_item = {equipment_pose_id: equipment_pose}
                equipment_poses[equipment_name].update(dict_item)

                if self.static_equipment:
                    # Ensure each pose is the same
                    first_key = list(equipment_poses[equipment_name].keys())[0]
                    assert np.allclose(equipment_pose, equipment_poses[equipment_name][first_key]), 'Equipment poses are not the same'
        return equipment_poses

    def parse_camera_poses(self):
        self.cam_transform = np.eye(4)
        self.cam_transform[1,1] = -1.0
        self.cam_transform[2,2] = -1.0
        camera_poses = {}
        camera_pose_paths = glob.glob(os.path.join(self.root, 'rgb_pose', '*.npy'))
        camera_pose_paths.sort()
        for camera_pose_path in camera_pose_paths:
            camera_pose = np.load(camera_pose_path)
            camera_pose = np.dot(camera_pose, self.cam_transform)
            camera_pose_id = int(os.path.basename(camera_pose_path).split('.')[0])
            dict_item = {camera_pose_id: camera_pose}
            camera_poses.update(dict_item)
        return camera_poses

    def parse_camera_intrinsics(self):
        camera_intrinsics_all = {}
        camera_intrinsics_paths = glob.glob(os.path.join(self.root, 'rgb_intrinsics', '*.npy'))
        camera_intrinsics_paths.sort()
        for camera_intrinsics_path in camera_intrinsics_paths:
            camera_intrinsics = np.load(camera_intrinsics_path)
            camera_intrinsics_id = int(os.path.basename(camera_intrinsics_path).split('.')[0])
            dict_item = {camera_intrinsics_id: camera_intrinsics}
            camera_intrinsics_all.update(dict_item)
        return camera_intrinsics_all

    def find_img_paths(self):
        img_paths = glob.glob(os.path.join(self.root, 'rgb', '*.jpg'))
        img_paths.sort()
        img_paths_dict = {}
        for img_path in img_paths:
            img_id = int(os.path.basename(img_path).split('.')[0])
            dict_item = {img_id: img_path}
            img_paths_dict.update(dict_item)
        return img_paths_dict

    def find_monodepth_paths(self):
        img_paths = glob.glob(os.path.join(self.root, 'monocular_depth', '*.png'))
        img_paths.sort()
        img_paths_dict = {}
        for img_path in img_paths:
            img_id = int(os.path.basename(img_path).split('.')[0])
            dict_item = {img_id: img_path}
            img_paths_dict.update(dict_item)
        return img_paths_dict

    def find_segmentation_paths(self):
        segmentation_paths = glob.glob(os.path.join(self.root, 'segmentation', '*.npy.gz'))
        segmentation_paths.sort()
        segmentation_paths_dict = {}
        for segmentation_path in segmentation_paths:
            segmentation_id = int(os.path.basename(segmentation_path).split('.')[0])
            dict_item = {segmentation_id: segmentation_path}
            segmentation_paths_dict.update(dict_item)
        return segmentation_paths_dict

    def find_equipment_point_paths(self):
        equipment_point_paths = glob.glob(os.path.join(self.root, 'equipment_points', '*.png'))
        equipment_point_paths.sort()
        equipment_point_paths_dict = {}
        for equipment_point_path in equipment_point_paths:
            equipment_point_id = int(os.path.basename(equipment_point_path).split('.')[0])
            dict_item = {equipment_point_id: equipment_point_path}
            equipment_point_paths_dict.update(dict_item)
        return equipment_point_paths_dict

    def find_depth_paths(self):
        depth_paths = glob.glob(os.path.join(self.root, 'depth', '*.png'))
        depth_paths.sort()
        depth_paths_dict = {}
        for depth_path in depth_paths:
            depth_id = int(os.path.basename(depth_path).split('.')[0])
            dict_item = {depth_id: depth_path}
            depth_paths_dict.update(dict_item)

        depth_bound_paths = glob.glob(os.path.join(self.root, 'depth', '*_bounds.npy'))
        depth_bound_paths.sort()
        depth_bound_paths_dict = {}
        for depth_bound_path in depth_bound_paths:
            depth_bound_id = int(os.path.basename(depth_bound_path).split('.')[0].replace('_bounds', ''))
            dict_item = {depth_bound_id: depth_bound_path}
            depth_bound_paths_dict.update(dict_item)

        return depth_paths_dict, depth_bound_paths_dict
    
    def find_assembly_paths(self):
        assembly_paths = glob.glob(os.path.join(self.root, 'assembly_preds', '*.npy'))
        assembly_paths.sort()
        assembly_paths_dict = {}
        for assembly_path in assembly_paths:
            assembly_id = int(os.path.basename(assembly_path).split('.')[0])
            dict_item = {assembly_id: assembly_path}
            assembly_paths_dict.update(dict_item)

        return assembly_paths_dict

    def exception_handler(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return []
        return wrapper

    @exception_handler
    def get_scene_dir_name(self):
        return self.root

    @exception_handler
    def get_present_equipment_names(self):
        return list(self.equipment_ids.keys())

    @exception_handler
    def get_equipment_id(self, equipment_name=None):
        if equipment_name is None:
            equipment_name = self.get_present_equipment_names()[0]
        return self.equipment_ids[equipment_name]

    @exception_handler
    def get_equipment_pose(self, equipment_name=None, img_idx=0):
        self.check_img_idx_arg(img_idx)
        if equipment_name is None:
            equipment_name = self.get_present_equipment_names()[0]
        if not self.equipment_poses[equipment_name]:
            equipment_name = 'root'
        return self.equipment_poses[equipment_name][img_idx]

    @exception_handler
    def get_camera_pose(self, img_idx):
        self.check_img_idx_arg(img_idx)
        return self.camera_poses[img_idx]

    @exception_handler
    def get_camera_intrinsics(self, img_idx):
        self.check_img_idx_arg(img_idx)
        return self.camera_intrinsics[img_idx]

    @exception_handler
    def get_number_of_images(self):
        return len(self.img_paths)

    @exception_handler
    def get_img_ids(self):
        return list(self.img_paths.keys())

    @exception_handler
    def get_equipment_to_camera(self, img_idx, equipment_name=None):
        self.check_img_idx_arg(img_idx)
        return np.dot(np.linalg.inv(self.get_camera_pose(img_idx)), self.get_equipment_pose(equipment_name, img_idx))

    @exception_handler
    def get_img(self, img_idx):
        self.check_img_idx_arg(img_idx)
        if img_idx not in self.imgs:
            img = cv2.imread(self.img_paths[img_idx])
            dict_item = {img_idx: img}
            self.imgs.update(dict_item)
        return self.imgs[img_idx]

    @exception_handler
    def get_monodepth(self, img_idx):
        self.check_img_idx_arg(img_idx)
        if img_idx not in self.monodepths:
            img = cv2.imread(self.monodepth_paths[img_idx])
            dict_item = {img_idx: img}
            self.monodepths.update(dict_item)
        return self.monodepths[img_idx]

    @exception_handler
    def get_segmentation(self, img_idx):
        self.check_img_idx_arg(img_idx)
        if img_idx not in self.segmentations:
            f = gzip.GzipFile(self.segmentation_paths[img_idx], "r")
            segmentation = np.load(f)
            dict_item = {img_idx: segmentation}
            self.segmentations.update(dict_item)
        return self.segmentations[img_idx]

    @exception_handler
    def get_equipment_points_norm(self, img_idx):
        self.check_img_idx_arg(img_idx)
        if img_idx not in self.equipment_points:
            equipment_point_img = cv2.imread(self.equipment_point_paths[img_idx])
            # Normalize to [0, 1]
            if equipment_point_img.dtype == np.uint8:
                equipment_point_img = equipment_point_img / 255.0
            elif equipment_point_img.dtype == np.uint16:
                equipment_point_img = equipment_point_img / 65535.0
            dict_item = {img_idx: equipment_point_img}
            self.equipment_points.update(dict_item)
        return self.equipment_points[img_idx]

    @exception_handler
    def get_segmentation_binary(self, img_idx, entity_ids):
        segmentation = self.get_segmentation(img_idx)
        segmentation_binary = np.zeros(segmentation.shape, dtype=np.uint8)
        for entity_id in entity_ids:
            segmentation_binary[segmentation == entity_id] = 1
        return segmentation_binary

    @exception_handler
    def get_segmentation_binary_full_extents(self, img_idx):
        equipment_point_img = self.get_equipment_points_norm(img_idx)
        return np.any(equipment_point_img > 0.0, axis=2)

    @exception_handler
    def get_depth(self, img_idx):
        ''' Returns depth in meters to the equipment'''
        self.check_img_idx_arg(img_idx)
        if img_idx not in self.depths:
            distance = cv2.imread(self.depth_paths[img_idx])
            distance = distance[:,:,0]
            depth_bounds = np.load(self.depth_bounds_paths[img_idx])
            distance = distance.astype(np.float32) / 255.0
            distance = distance * (depth_bounds[1] - depth_bounds[0])

            # convert from distance image to depth image
            intrinsics = self.get_camera_intrinsics(img_idx)
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]

            height, width = distance.shape
            xlin = np.linspace(0, width - 1, width)
            ylin = np.linspace(0, height - 1, height)
            px, py = np.meshgrid(xlin, ylin)

            x_over_z = (px - cx) / fx
            y_over_z = (py - cy) / fy

            depth = distance / np.sqrt(1. + x_over_z**2 + y_over_z**2)

            dict_item = {img_idx: depth}
            self.depths.update(dict_item)
        return self.depths[img_idx]

    def check_img_idx_arg(self, img_idx):
        assert img_idx in self.get_img_ids(), 'img_idx out of range'

    @exception_handler
    def get_real_assembly_preds(self, img_idx):
        ''' Returns ground truth predicted assemblies'''
        self.check_img_idx_arg(img_idx)
        if img_idx not in self.assembly_predictions:
            assembly_pred = np.load(self.assembly_paths[img_idx])
            dict_item = {img_idx: assembly_pred}
            self.assembly_predictions.update(dict_item)
        return self.assembly_predictions[img_idx]
    
    @exception_handler
    def get_visible_parts(self, img_idx):
        self.check_img_idx_arg(img_idx)
        try:
            # Get part colors
            with open(os.path.join(self.root, 'entity_ids.yaml')) as f:
                equipment_data = yaml.safe_load(f)

            # Get observed colors
            seg_dir = self.segmentation_paths[img_idx]            
            seg_file = gzip.GzipFile(seg_dir)
            img = np.load(seg_file)
            all_unique_colors = (np.unique(img)).tolist()

            # Make visible part label (corresponding to assembly state label)
            visible_parts = [k for k, v in equipment_data.items() if v in all_unique_colors]

            return visible_parts
        except:
            return [] # This means None (return empty dict)

    def get_absent_but_observable_parts(self, img_idx):
        ''' Returns parts that are absent but would be visible if they were present'''
        self.check_img_idx_arg(img_idx)
        try:
            yaml_file = os.path.join(self.root, 'observable_parts', 'absent_but_would_be_visible_' + str(img_idx).zfill(5) + '.yaml')
            with open(yaml_file) as f:
                absent_but_observable_parts = yaml.safe_load(f)
            return absent_but_observable_parts
        except:
            return None

    def get_present_and_observable_parts(self, img_idx):
        ''' Returns parts that are present and visible'''
        self.check_img_idx_arg(img_idx)
        try:
            yaml_file = os.path.join(self.root, 'observable_parts', 'present_and_visible_' + str(img_idx).zfill(5) + '.yaml')
            with open(yaml_file) as f:
                absent_but_observable_parts = yaml.safe_load(f)
            return absent_but_observable_parts
        except:
            return None
        
    def set_part_names(self, part_names):
        self.part_names = part_names

    def get_absent_and_not_observable_parts(self, img_idx):
        ''' Returns parts that are absent and would not be visible if they were present'''
        self.check_img_idx_arg(img_idx)
        if self.part_names == []:
            raise ValueError('Part names must be provided')
        absent_but_observable = set(self.get_absent_but_observable_parts(img_idx))
        all_parts = set(self.part_names)
        present_parts = set(self.get_present_equipment_names())
        absent_parts = all_parts - present_parts
        absent_and_not_observable_parts = list(absent_parts - absent_but_observable)
        return absent_and_not_observable_parts
    
    def get_present_and_not_observable_parts(self, img_idx):
        ''' Returns parts that are present and not visible'''
        self.check_img_idx_arg(img_idx)
        if self.part_names == []:
            raise ValueError('Part names must be provided')
        present_and_observable = set(self.get_present_and_observable_parts(img_idx))
        present_parts = set(self.get_present_equipment_names())
        present_and_not_observable_parts = list(present_parts - present_and_observable)
        return present_and_not_observable_parts
    
    def part_names_to_binary_array(self, part_names):
        binary_array = np.zeros(len(self.part_names))
        if self.part_names == []:
            raise ValueError('Part names must be provided')
        binary_array = np.zeros(len(self.part_names))
        for i in range(len(self.part_names)):
            if self.part_names[i] in part_names:
                binary_array[i] = 1
        return binary_array


class DatasetPhaseNvisii():
    def __init__(self, path, phase, static_equipment=True, real_assemblies=False):
        self.root = path
        self.phase = phase
        self.static_equipment = static_equipment
        self.real_assemblies = real_assemblies
        self.scene_dirs = glob.glob(os.path.join(self.root, phase) + '/*')
        self.scene_dirs.sort()

        # Load object meshes
        self.mesh_paths = glob.glob(os.path.join(self.root, 'models/*.obj'))
        self.mesh_paths.sort()

        self.check_for_error()

        self.meshes = {}
        self.meshes_trimesh = {}
        self.meshes_sampled_points = {}
        self.mesh_centroids = {}
        self.mesh_names = []
        mesh_min_bounds = []
        mesh_max_bounds = []
        for mesh_path in self.mesh_paths:
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            mesh_name = os.path.basename(mesh_path).split('.obj')[0]
            if mesh.has_vertices():
                self.mesh_names.append(mesh_name)
                mesh.compute_vertex_normals()
                self.mesh_centroids[mesh_name] = mesh.get_center()
                mesh_min_bounds.append(mesh.get_min_bound())
                mesh_max_bounds.append(mesh.get_max_bound())
                points = np.asarray(mesh.sample_points_uniformly(100).points)
                self.meshes_sampled_points[mesh_name] = np.hstack((points, np.ones((100, 1)))).T

                trimesh_mesh = trimesh.load(mesh_path)
                self.meshes_trimesh[mesh_name] = trimesh_mesh

        self.mesh_min_bounds = np.array(mesh_min_bounds)
        self.mesh_max_bounds = np.array(mesh_max_bounds)
        self.overall_mesh_min_bounds = np.min(mesh_min_bounds, axis=0)
        self.overall_mesh_max_bounds = np.max(mesh_max_bounds, axis=0)

    def check_for_error(self):
        if not os.path.exists(self.root):
            raise FileNotFoundError("Dataset root does not exist: " + self.root)
        elif not os.path.exists(os.path.join(self.root, self.phase)):
            raise FileNotFoundError("Dataset phase does not exist: " + self.phase)
        elif len(self.scene_dirs) == 0:
            raise FileNotFoundError("No scenes found in: " + os.path.join(self.root, self.phase))
        
        if len(self.mesh_paths) == 0:
            raise FileNotFoundError("No obj meshes found in: " + os.path.join(self.root, 'models/*.obj'))

    def num_scenes(self):
        return len(self.scene_dirs)

    def get_scene(self, scene_idx, labeled=True):
        return NvisiiScene(self.scene_dirs[scene_idx], static_equipment=self.static_equipment, labeled=labeled, real_assemblies=self.real_assemblies)

    def compute_max_error(self, T_est, T_act, mesh_name=None):
        if mesh_name is not None:
            mesh_points = self.meshes_sampled_points[mesh_name]
        else:
            mesh_points = self.get_overall_sampled_points()
        
        T_est_points = np.dot(T_est, mesh_points)
        T_act_points = np.dot(T_act, mesh_points)
        error = np.linalg.norm(T_est_points - T_act_points, axis=0)
        max_error = np.max(error).astype(np.float32)
        return max_error

    def get_mesh(self, mesh_name=None):
        if mesh_name is None:
            mesh_name = self.mesh_names[0]
        if mesh_name in self.meshes.keys():
            return self.meshes[mesh_name]
        else:
            mesh = o3d.io.read_triangle_mesh(os.path.join(self.root, 'models', mesh_name + '.obj'))
            self.meshes[mesh_name] = mesh
            return mesh

    def get_trimesh(self, mesh_name=None):
        if mesh_name is None:
            mesh_name = self.mesh_names[0]
        return self.meshes_trimesh[mesh_name]

    def get_mesh_names(self):
        return self.mesh_names

    def get_mesh_sampled_points(self, mesh_name=None):
        if mesh_name is None:
            mesh_name = self.mesh_names[0]
        return self.meshes_sampled_points[mesh_name]
    
    def get_mesh_centroids(self):
        return self.mesh_centroids

    def get_overall_sampled_points(self):
        sampled_points = []
        for mesh_name in self.meshes_sampled_points.keys():
            sampled_points.append(self.meshes_sampled_points[mesh_name])
        return np.hstack(sampled_points)

    def project_points_and_get_bounding_box(self, object_to_cam, intrinsics, width, height, pad_factor=0.0):
        object_points = self.get_overall_sampled_points()

        # Transform object points to camera coordinates
        cam_points = np.dot(object_to_cam, object_points)
        # Divide by the homogeneous coordinate to get non-homogeneous coordinates
        cam_points = cam_points[:3] / cam_points[3]
        # Project points to image plane using camera intrinsics
        img_points = np.dot(intrinsics, cam_points)
        # Divide by the z-coordinate to get pixel coordinates
        img_points = img_points / img_points[2]
        # Get bounding box of the projected points
        x_min, y_min = np.min(img_points[:2], axis=1)
        x_max, y_max = np.max(img_points[:2], axis=1)
        # Pad the bounding box
        x_min -= pad_factor * (x_max - x_min)
        y_min -= pad_factor * (y_max - y_min)
        x_max += pad_factor * (x_max - x_min)
        y_max += pad_factor * (y_max - y_min)
        # Clamp to image dimensions
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width-1, x_max)
        y_max = min(height-1, y_max)
        return np.asarray([x_min, y_min, x_max, y_max])


def validate_transforms(equipment_renderer, equipment_to_camera):
    color, depth = equipment_renderer.render_object_pose(equipment_to_camera)
    return color

