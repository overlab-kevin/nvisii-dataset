import os
import sys
import argparse
import numpy as np
import cv2

# Ensure project root on sys.path
PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from nvisii_dataset.interface_nvisii import DatasetPhaseNvisii  # noqa: E402


def _project_points(K, T_obj_cam, V):
    """
    Project 3D vertices V (N,3) in object frame into pixels using camera intrinsics K (3x3)
    and object-to-camera transform T_obj_cam (4x4).
    Returns:
      uv: (N,2) float pixel coords
      z: (N,) depth in camera coords
      mask: (N,) bool, True if z>0
    """
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError("Vertices must be (N,3)")
    N = V.shape[0]
    V_h = np.hstack([V, np.ones((N, 1), dtype=V.dtype)])  # (N,4)
    Pc = (T_obj_cam @ V_h.T)  # (4,N)
    X, Y, Z, W = Pc[0], Pc[1], Pc[2], Pc[3]
    # Homogeneous normalization (should be 1)
    eps = 1e-8
    X /= np.where(np.abs(W) < eps, 1.0, W)
    Y /= np.where(np.abs(W) < eps, 1.0, W)
    Z /= np.where(np.abs(W) < eps, 1.0, W)
    mask = Z > 0.0

    XYZ = np.vstack([X, Y, Z])  # (3,N)
    UVW = (K @ XYZ)             # (3,N)
    u = UVW[0] / np.maximum(eps, UVW[2])
    v = UVW[1] / np.maximum(eps, UVW[2])
    uv = np.stack([u, v], axis=1)  # (N,2)
    return uv, Z, mask


def _draw_wireframe(img, uv, faces, z, vis_mask, color=(0, 255, 0), line_w=1, tri_step=1):
    """
    Draw triangle wireframe on img given projected vertices uv (N,2), faces (M,3),
    per-vertex depth z (N,), and visibility mask vis_mask (N,).
    tri_step: subsample every k-th triangle to speed up drawing on dense meshes.
    """
    h, w = img.shape[:2]
    # Convert to int coords
    uv_i = uv.astype(np.int32)

    # Cull triangles if any vertex behind camera
    fidx = np.arange(faces.shape[0])[::max(1, tri_step)]
    for fi in fidx:
        f = faces[fi]
        if not (vis_mask[f[0]] and vis_mask[f[1]] and vis_mask[f[2]]):
            continue
        pts = uv_i[f].reshape(-1, 1, 2)
        # Optional: further cull triangles far outside the image bounds quickly
        if (pts[:, 0, 0].max() < 0) or (pts[:, 0, 1].max() < 0) or (pts[:, 0, 0].min() >= w) or (pts[:, 0, 1].min() >= h):
            continue
        # Draw edges
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=line_w, lineType=cv2.LINE_AA)


def _color_from_name(name):
    """
    Deterministic BGR color from string name.
    """
    h = abs(hash(name))
    b = 64 + (h % 192)
    g = 64 + ((h // 192) % 192)
    r = 64 + ((h // (192 * 192)) % 192)
    return int(b % 256), int(g % 256), int(r % 256)


def visualize_dataset(dataset_root, phase, samples_per_scene=3, wait=0, line_w=1, tri_step=1):
    ds = DatasetPhaseNvisii(dataset_root, phase, static_equipment=True, real_assemblies=False)
    n_scenes = ds.num_scenes()
    print(f"Loaded dataset: {dataset_root} | phase: {phase} | scenes: {n_scenes}")
    print("Instruction: press any key to advance. Use 'q' or Esc to quit.")

    cv2.namedWindow("RGB + Mesh Wireframe", cv2.WINDOW_NORMAL)

    for s_idx in range(n_scenes):
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
        img_ids = sorted(img_ids)[:max(0, samples_per_scene)]

        try:
            present_names = list(scene.get_present_equipment_names() or [])
        except Exception:
            present_names = []

        for img_id in img_ids:
            rgb = scene.get_img(img_id)
            if not isinstance(rgb, np.ndarray) or rgb.size == 0:
                print(f"[{scene_name}] img {img_id}: failed to load RGB.")
                continue

            K = scene.get_camera_intrinsics(img_id)
            if not (isinstance(K, np.ndarray) and K.shape == (3, 3)):
                print(f"[{scene_name}] img {img_id}: invalid intrinsics.")
                continue

            overlay = rgb.copy()
            h, w = overlay.shape[:2]

            for name in present_names:
                # Get mesh geometry
                try:
                    tri = ds.get_trimesh(name)
                    V = np.asarray(tri.vertices, dtype=np.float32)
                    F = np.asarray(tri.faces, dtype=np.int32)
                    if V.size == 0 or F.size == 0:
                        continue
                except Exception:
                    # Fallback to Open3D mesh if available
                    try:
                        m = ds.get_mesh(name)
                        if not m.has_vertices() or not m.has_triangles():
                            continue
                        V = np.asarray(m.vertices, dtype=np.float32)
                        F = np.asarray(m.triangles, dtype=np.int32)
                    except Exception:
                        continue

                # Pose: object -> camera
                try:
                    T_obj_cam = scene.get_equipment_to_camera(img_id, name)
                    if not (isinstance(T_obj_cam, np.ndarray) and T_obj_cam.shape == (4, 4)):
                        continue
                except Exception:
                    continue

                # Project and draw
                try:
                    uv, z, mask = _project_points(K, T_obj_cam, V)
                    color = _color_from_name(name)
                    _draw_wireframe(overlay, uv, F, z, mask, color=color, line_w=line_w, tri_step=max(1, tri_step))
                except Exception as e:
                    # Skip problematic meshes but continue visualization
                    # print(f"Projection failed for '{name}' on img {img_id}: {e}")
                    continue

            title = f"[{phase}] {scene_name} | img {img_id}"
            cv2.imshow("RGB + Mesh Wireframe", overlay)
            key = cv2.waitKey(wait) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                cv2.destroyAllWindows()
                return

    print("Done.")
    cv2.destroyAllWindows()


def parse_args():
    ap = argparse.ArgumentParser(description="Visualize dataset: overlay present meshes onto RGB using intrinsics/extrinsics.")
    ap.add_argument("dataset", help="Path to dataset root directory")
    ap.add_argument("phase", help="Dataset phase subfolder (e.g., train, val, test)")
    ap.add_argument("--samples-per-scene", type=int, default=3, help="How many images to visualize per scene")
    ap.add_argument("--wait", type=int, default=0, help="waitKey delay in ms (0=wait for key)")
    ap.add_argument("--line-width", type=int, default=1, help="Wireframe line width in pixels")
    ap.add_argument("--tri-step", type=int, default=1, help="Draw every k-th triangle for speed (>=1)")
    return ap.parse_args()


def main():
    args = parse_args()
    visualize_dataset(
        dataset_root=args.dataset,
        phase=args.phase,
        samples_per_scene=max(1, args.samples_per_scene),
        wait=args.wait,
        line_w=max(1, args.line_width),
        tri_step=max(1, args.tri_step),
    )


if __name__ == "__main__":
    main()
