import os
import sys
if sys.platform == 'win32':
    from pyglet.gl.wgl import *  # noqa: F401,F403
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'
else:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

import argparse
import numpy as np
import cv2

# Ensure project root on sys.path
PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from pyrender import IntrinsicsCamera, Mesh, Scene, OffscreenRenderer, Node, DirectionalLight  # noqa: E402
from nvisii_dataset.interface_nvisii import DatasetPhaseNvisii  # noqa: E402


class EquipmentRenderer:
    def __init__(self):
        self._initialized = False
        self.width = None
        self.height = None
        self.renderer = None
        self.scene = None
        self.mesh_nodes = {}
        self.cam_node = None
        self.light_node = None

    def release(self):
        try:
            if self.renderer is not None:
                self.renderer.delete()
        except Exception:
            pass
        self._initialized = False
        self.width = None
        self.height = None
        self.renderer = None
        self.scene = None
        self.mesh_nodes = {}
        self.cam_node = None
        self.light_node = None

    def init(self, width, height, mesh_trimeshes_dict, intrinsics):
        # Recreate renderer/scene if viewport changes or not initialized
        if (not self._initialized) or (width != self.width) or (height != self.height):
            self.release()
            self.width, self.height = int(width), int(height)
            self.renderer = OffscreenRenderer(viewport_width=self.width, viewport_height=self.height)
            self.scene = Scene()
            self.mesh_nodes = {}
            for mesh_name, mesh_trimesh in mesh_trimeshes_dict.items():
                try:
                    mesh = Mesh.from_trimesh(mesh_trimesh)
                    node = Node(mesh=mesh)
                    self.mesh_nodes[mesh_name] = node
                    self.scene.add_node(node)
                except Exception:
                    continue

            cam_pose = np.eye(4, dtype=np.float32)
            cam_pose[1, 1] = -1.0
            cam_pose[2, 2] = -1.0
            cam = IntrinsicsCamera(intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2])
            self.cam_node = self.scene.add(cam, pose=cam_pose)

            light = DirectionalLight(color=np.ones(3), intensity=50.0)
            self.light_node = self.scene.add(light, pose=cam_pose)
            self._initialized = True
        else:
            # Only intrinsics may have changed
            self.set_intrinsics(intrinsics)

    def set_intrinsics(self, intrinsics):
        if not self._initialized:
            raise RuntimeError("Renderer not initialized")
        # Replace camera with updated intrinsics
        try:
            if self.cam_node is not None:
                self.scene.remove_node(self.cam_node)
        except Exception:
            pass
        cam_pose = np.eye(4, dtype=np.float32)
        cam_pose[1, 1] = -1.0
        cam_pose[2, 2] = -1.0
        cam = IntrinsicsCamera(intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2])
        self.cam_node = self.scene.add(cam, pose=cam_pose)

    def set_visibilities(self, visible_mesh_names):
        if not self._initialized:
            raise RuntimeError("Renderer not initialized")
        visible = set(visible_mesh_names or [])
        for name, node in self.mesh_nodes.items():
            if node.mesh is not None:
                node.mesh.is_visible = name in visible

    def render_with_poses(self, obj_to_cam_map):
        if not self._initialized:
            raise RuntimeError("Renderer not initialized")
        # Set per-mesh poses and visibility
        for name, node in self.mesh_nodes.items():
            T = obj_to_cam_map.get(name, None)
            if T is None:
                if node.mesh is not None:
                    node.mesh.is_visible = False
                continue
            self.scene.set_pose(node, T)
        color, depth = self.renderer.render(self.scene)
        return color, depth


def _rgb_from_pyrender(color_img):
    # pyrender returns RGB or RGBA uint8; convert to BGR for OpenCV
    if color_img.ndim == 3 and color_img.shape[2] == 4:
        return cv2.cvtColor(color_img, cv2.COLOR_RGBA2BGR)
    elif color_img.ndim == 3 and color_img.shape[2] == 3:
        return cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    else:
        # Fallback
        return cv2.cvtColor(color_img, cv2.COLOR_GRAY2BGR)


def visualize_dataset(dataset_root, phase, samples_per_scene=3, wait=0, seed=None):
    ds = DatasetPhaseNvisii(dataset_root, phase, static_equipment=True, real_assemblies=False)
    n_scenes = ds.num_scenes()
    print(f"Loaded dataset: {dataset_root} | phase: {phase} | scenes: {n_scenes}")
    print("Instruction: press any key to advance scenes. Use 'q' or Esc to quit.")
    rng = np.random.RandomState(None if seed is None else int(seed))

    cv2.namedWindow("Scene Samples (RGB top | Render bottom)", cv2.WINDOW_NORMAL)

    for s_idx in range(n_scenes):
        # Load scene and metadata
        try:
            scene = ds.get_scene(s_idx, labeled=False)
            scene_dir = scene.get_scene_dir_name()
            scene_name = os.path.basename(scene_dir) if isinstance(scene_dir, str) else f"scene_{s_idx}"
        except Exception as e:
            print(f"[Scene {s_idx}] failed to load: {e}")
            continue

        img_ids = scene.get_img_ids()
        if not isinstance(img_ids, list) or len(img_ids) == 0:
            print(f"[{scene_name}] No images found.")
            continue
        img_ids = sorted(img_ids)

        k = min(max(1, int(samples_per_scene)), len(img_ids))
        sel = rng.choice(img_ids, size=k, replace=False)

        try:
            present_names = list(scene.get_present_equipment_names() or [])
        except Exception:
            present_names = []

        # Prepare trimesh dict only for present objects
        mesh_trimeshes = {}
        for name in present_names:
            try:
                mesh_trimeshes[name] = ds.get_trimesh(name)
            except Exception:
                pass
        if len(mesh_trimeshes) == 0:
            print(f"[{scene_name}] No present meshes to render.")
            continue

        renderer = EquipmentRenderer()
        stacks = []
        base_wh = None

        for img_id in sel:
            rgb = scene.get_img(img_id)
            if not isinstance(rgb, np.ndarray) or rgb.size == 0:
                print(f"[{scene_name}] img {img_id}: failed to load RGB.")
                continue
            h, w = rgb.shape[:2]

            K = scene.get_camera_intrinsics(img_id)
            if not (isinstance(K, np.ndarray) and K.shape == (3, 3)):
                print(f"[{scene_name}] img {img_id}: invalid intrinsics.")
                continue

            # Initialize/Update renderer for this image size and intrinsics
            renderer.init(w, h, mesh_trimeshes, K)
            renderer.set_visibilities(present_names)

            # Build per-mesh object->camera pose map
            obj_to_cam = {}
            for name in present_names:
                try:
                    T = scene.get_equipment_to_camera(img_id, name)
                    if isinstance(T, np.ndarray) and T.shape == (4, 4):
                        obj_to_cam[name] = T.astype(np.float32)
                except Exception:
                    continue

            # Render and convert for OpenCV display
            color, _ = renderer.render_with_poses(obj_to_cam)
            render_bgr = _rgb_from_pyrender(color)

            # Ensure same size as RGB for stacking
            if render_bgr.shape[:2] != (h, w):
                render_bgr = cv2.resize(render_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

            stack = np.vstack([rgb, render_bgr])
            stacks.append(stack)
            if base_wh is None:
                base_wh = (stack.shape[1], stack.shape[0])

        # Clean up renderer resources between scenes
        renderer.release()

        if len(stacks) == 0:
            print(f"[{scene_name}] No valid samples to display.")
            continue

        # Align widths/heights if needed and concatenate horizontally
        target_h = max(s.shape[0] for s in stacks)
        target_w_each = max(s.shape[1] for s in stacks)
        normed = []
        for s in stacks:
            if s.shape[0] != target_h or s.shape[1] != target_w_each:
                s = cv2.resize(s, (target_w_each, target_h), interpolation=cv2.INTER_LINEAR)
            normed.append(s)
        canvas = cv2.hconcat(normed)

        title = f"[{phase}] {scene_name} | {len(stacks)} sample(s)"
        cv2.imshow("Scene Samples (RGB top | Render bottom)", canvas)
        key = cv2.waitKey(wait if wait > 0 else 0) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            cv2.destroyAllWindows()
            return

    print("Done.")
    cv2.destroyAllWindows()


def parse_args():
    ap = argparse.ArgumentParser(description="Visualize dataset: per-scene, show random RGBs (top) with rendered meshes (bottom).")
    ap.add_argument("dataset", help="Path to dataset root directory")
    ap.add_argument("phase", help="Dataset phase subfolder (e.g., train, val, test)")
    ap.add_argument("--samples-per-scene", type=int, default=3, help="How many images to visualize per scene (randomly sampled)")
    ap.add_argument("--wait", type=int, default=0, help="waitKey delay in ms per scene (0=wait for key)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling")
    return ap.parse_args()


def main():
    args = parse_args()
    visualize_dataset(
        dataset_root=args.dataset,
        phase=args.phase,
        samples_per_scene=max(1, args.samples_per_scene),
        wait=args.wait,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
