#!/usr/bin/env python3
"""
Create Rendering Script for GEN3C (Multiview Waymo Input)
This script mirrors create_rendering_multiview_image_input.py structure and flow,
but reads multiview data from a Waymo-style folder containing:
  images/{camera}.png
  pose/{camera}.npy         (4x4 world2camera, OpenCV)
  intrinsics/{camera}.npy   (3x3)
  mask/{camera}.npy         (HxW ones)
  lidar/{camera}.npy        (HxW meters, 0 where no depth)
"""

import argparse
import os
import cv2
import torch
import numpy as np
from typing import Optional, Tuple, List
import torch.nn.functional as F
import json
import sys
from pathlib import Path
from tqdm import tqdm

# Import necessary modules from the original codebase
from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_BufferSelector
from cosmos_predict1.diffusion.inference.camera_utils import generate_camera_trajectory
from cosmos_predict1.diffusion.inference.camera_sequence_generation import (
    generate_source_to_target_trajectory, 
    load_map_to_camera_tf_matrix, 
    generate_pixel_focused_trajectory
)
from cosmos_predict1.utils import log, misc
from cosmos_predict1.diffusion.inference.inference_utils import (
    add_common_arguments,
)

# Alignment helper (for optional depth alignment to LiDAR)
from cosmos_predict1.diffusion.inference.camera_utils import _align_inv_depth_to_depth

torch.enable_grad(False)

def _resize_sparse_depth(depth_map: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Resize sparse depth map by creating empty target and carefully placing rescaled depth values.
    This preserves the exact sparse structure without interpolation artifacts.
    Handles index collisions by averaging overlapping depth values.
    
    Args:
        depth_map: Input depth map with sparse valid values (0 = invalid)
        target_shape: Target (height, width) tuple
        
    Returns:
        Resized depth map with preserved sparse structure
    """
    target_h, target_w = target_shape
    orig_h, orig_w = depth_map.shape
    
    # Create empty depth map at target size
    resized_depth = np.zeros(target_shape, dtype=np.float32)
    
    # Find indices with non-zero depth values
    valid_indices = np.where(depth_map > 0)
    if len(valid_indices[0]) == 0:
        return resized_depth  # No valid depth values
    
    # Get valid depth values
    valid_depths = depth_map[valid_indices]
    
    # Calculate scaling factors
    scale_h = target_h / orig_h
    scale_w = target_w / orig_w
    
    # Rescale the indices to target size
    rescaled_y = np.round(valid_indices[0] * scale_h).astype(int)
    rescaled_x = np.round(valid_indices[1] * scale_w).astype(int)
    
    # Clamp indices to valid range
    rescaled_y = np.clip(rescaled_y, 0, target_h - 1)
    rescaled_x = np.clip(rescaled_x, 0, target_w - 1)
    
    # Place the depth values at the rescaled indices
    resized_depth[rescaled_y, rescaled_x] = valid_depths
    
    return resized_depth

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the rendering script (mirrors image_input)."""
    parser = argparse.ArgumentParser(description="Create rendered warp images and masks from Waymo multiview folder")
    # Add common arguments
    add_common_arguments(parser)

    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Waymo-exported folder containing images/, pose/, intrinsics/, mask/, lidar/"
    )

    parser.add_argument(
        "--trajectory",
        type=str,
        choices=[
            "left", "right", "up", "down", "zoom_in", "zoom_out",
            "clockwise", "counterclockwise", "none"
        ],
        default=None,
        help="Select a trajectory type from the available options"
    )
    parser.add_argument(
        "--camera_rotation",
        type=str,
        choices=["center_facing", "no_rotation", "trajectory_aligned"],
        default=None,
        help="Controls camera rotation during movement"
    )
    parser.add_argument(
        "--movement_distance",
        type=float,
        default=None,
        help="Distance of the camera from the center of the scene"
    )

    parser.add_argument(
        "--noise_aug_strength",
        type=float,
        default=0.0,
        help="Strength of noise augmentation on warped frames"
    )

    parser.add_argument(
        "--filter_points_threshold",
        type=float,
        default=0.05,
        help="If set, filter the points continuity of the warped images.",
    )

    parser.add_argument(
        "--foreground_masking",
        action="store_true",
        help="If set, use foreground masking for the warped images.",
    )
    
    parser.add_argument(
        "--rendered_tensor_dir",
        type=str,
        default="../test_results/rendered_tensor_dir",
        help="Directory to save rendered tensors"
    )
    
    parser.add_argument(
        "--rendered_images_path",
        type=str,
        required=True,
        help="Filename for the rendered warp images tensor"
    )
    
    parser.add_argument(
        "--rendered_masks_path",
        type=str,
        required=True,
        help="Filename for the rendered warp masks tensor"
    )
    
    parser.add_argument(
        "--trajectory_generation_method",
        type=str,
        choices=["action_based_movement", "pixel_focusing", "source_to_target_linear_interpolation"],
        required=True,
        help="Method for generating camera trajectory: action_based_movement, pixel_focusing, or source_to_target_linear_interpolation"
    )
    
    parser.add_argument(
        "--target_pixel_x",
        type=int,
        default=None,
        help="Target pixel X coordinate for pixel focusing method"
    )
    
    parser.add_argument(
        "--target_pixel_y",
        type=int,
        default=None,
        help="Target pixel Y coordinate for pixel focusing method"
    )
    
    parser.add_argument(
        "--movement_ratio",
        type=float,
        default=None,
        help="Movement ratio (0-1) for pixel focusing method"
    )
    
    parser.add_argument(
        "--start_transition_frames",
        type=int,
        default=None,
        help="Frame number to start transitioning to target for pixel focusing method"
    )
    
    parser.add_argument(
        "--end_transition_frames",
        type=int,
        default=None,
        help="Frame number to end transitioning to target for pixel focusing method"
    )
    
    parser.add_argument(
        "--source_pose_path",
        type=str,
        default=None,
        help="Source pose path (for source_to_target_linear_interpolation); for Waymo, pass pose/NAME.npy"
    )
    
    parser.add_argument(
        "--target_pose_path",
        type=str,
        default=None,
        help="Target pose path (for source_to_target_linear_interpolation); for Waymo, pass pose/NAME.npy"
    )

    # Waymo-specific optional alignment
    parser.add_argument(
        "--align_depth_with_lidar",
        action="store_true",
        help="If set, align predicted depth to LiDAR depth using valid LiDAR mask"
    )
    
    # Missing arguments that are referenced in the code
    parser.add_argument(
        "--reference_frame",
        type=int,
        default=0,
        help="Index of the reference frame to use for trajectory generation (default: 0)"
    )
    
    parser.add_argument(
        "--frame_buffer_max",
        type=int,
        default=2,
        help="Maximum number of frames to keep in buffer for Cache3D_BufferSelector"
    )
    
    
    
    return parser


def _list_cameras(waymo_root: Path) -> List[str]:
    images_dir = waymo_root / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images folder: {images_dir}")
    cams = sorted([p.stem for p in images_dir.glob("*.png")])
    if not cams:
        raise RuntimeError(f"No images found in {images_dir}")
    return cams


def create_rendering(args) -> None:
    """
    Create rendering from Waymo multiview input folder.
    Mirrors the flow of create_rendering_multiview_image_input.py.
    """
    misc.set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup DepthAnythingV2 (import location mirrors image_input)
    base = Path.cwd().parent.parent
    sys.path.insert(0, str(base / "Depth-Estimation" / "Depth-Anything-V2" / "metric_depth"))
    from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    encoder = 'vitl'
    max_depth = 80
    DepthAnythingV2_checkpoint_path = base / 'Depth-Estimation' / 'Depth-Anything-V2' / 'checkpoints' / f'depth_anything_v2_metric_hypersim_{encoder}.pth'
    
    dav2_model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    dav2_model.load_state_dict(torch.load(DepthAnythingV2_checkpoint_path, map_location=device))
    dav2_model.to(device).eval()
    
    log.info(f"Loading DepthAnythingV2 model from {DepthAnythingV2_checkpoint_path}")
    
    # Load input Waymo folder
    input_folder = Path(args.input_folder)
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder {input_folder} does not exist")

    cameras = _list_cameras(input_folder)
    N = len(cameras)
    log.info(f"Found {N} camera views in {input_folder}")
    
    input_images = torch.zeros(N, 3, args.height, args.width)  # [N, C, H, W]
    input_depths = torch.zeros(N, 1, args.height, args.width)  # [N, 1, H, W]
    input_masks = torch.ones(N, 1, args.height, args.width)    # [N, 1, H, W]
    input_w2cs = torch.zeros(N, 4, 4)                          # [N, 4, 4]
    input_intrinsics = torch.zeros(N, 3, 3)                     # [N, 3, 3]

    log.info("Starting depth estimation and data assembly from Waymo folder...")
    with torch.no_grad():
        for i, cam in enumerate(tqdm(cameras, desc="Processing views")):
            # Pose (W2C)
            pose = np.load(input_folder / "pose" / f"{cam}.npy").astype(np.float32)
            input_w2cs[i] = torch.from_numpy(pose)

            # Image - load once and get dimensions
            img_path = input_folder / "images" / f"{cam}.png"
            img = cv2.imread(str(img_path))
            actual_height, actual_width = img.shape[:2]
            img_resized = cv2.resize(img, (args.width, args.height))
            
            # Load and scale intrinsics based on this image's actual dimensions
            K = np.load(input_folder / "intrinsics" / f"{cam}.npy").astype(np.float32)
            scale_h = args.height / actual_height
            scale_w = args.width / actual_width
            K[0, 0] *= scale_w  # fx
            K[1, 1] *= scale_h  # fy
            K[0, 2] *= scale_w  # cx
            K[1, 2] *= scale_h  # cy
            input_intrinsics[i] = torch.from_numpy(K)

            # DepthAnythingV2 depth
            depth = dav2_model.infer_image(img_resized)

            # Optional LiDAR alignment
            if args.align_depth_with_lidar:
                lidar_path = input_folder / "lidar" / f"{cam}.npy"
                if lidar_path.exists():
                    lidar_depth = np.load(lidar_path).astype(np.float32)
                    if lidar_depth.shape != (args.height, args.width):
                        # Use sparse-aware resizing for LiDAR data
                        lidar_depth = _resize_sparse_depth(lidar_depth, (args.height, args.width))
                    pred_t = torch.from_numpy(depth).float()
                    lidar_t = torch.from_numpy(lidar_depth).float()
                    target_mask = lidar_t > 0
                    depth_aligned = _align_inv_depth_to_depth(
                        1.0 / torch.clamp_min(pred_t, 1e-6),
                        lidar_t,
                        target_mask=target_mask
                    )
                    depth = depth_aligned.numpy()

            # Convert BGR->RGB and normalize
            frame_rgb = img_resized[..., [2, 1, 0]].copy()
            frame_tensor = torch.from_numpy(frame_rgb).float()
            frame_tensor = frame_tensor.permute(2, 0, 1)
            frame_tensor = frame_tensor / 127.5 - 1.0

            # Masks - resize to target dimensions if needed
            mask_path = input_folder / "mask" / f"{cam}.npy"
            if mask_path.exists():
                mask_np = np.load(mask_path).astype(np.float32)
                if mask_np.shape != (args.height, args.width):
                    mask_np = cv2.resize(mask_np, (args.width, args.height), interpolation=cv2.INTER_NEAREST)
                mask_t = torch.from_numpy(mask_np)
            else:
                mask_t = torch.ones(args.height, args.width, dtype=torch.float32)

            # Store
            input_images[i] = frame_tensor
            input_depths[i, 0] = torch.from_numpy(depth).float()
            input_masks[i, 0] = mask_t

            # Clear intermediates
            del img_resized, frame_rgb, frame_tensor, depth, img, pose
            torch.cuda.empty_cache()

    # Model cleanup
    del dav2_model
    torch.cuda.empty_cache()
    log.info("Cleared depth model and intermediates from memory")

    # Move to device
    input_images = input_images.to(device)
    input_depths = input_depths.to(device)
    input_masks = input_masks.to(device)
    input_w2cs = input_w2cs.to(device)
    input_intrinsics = input_intrinsics.to(device)

    # Add batch dimension for Cache3D_BufferSelector: [1, N, C, H, W]
    input_images_bNCHW = input_images.unsqueeze(0)
    input_depths_bN1HW = input_depths.unsqueeze(0)
    input_masks_bN1HW = input_masks.unsqueeze(0)
    input_w2cs_bN44 = input_w2cs.unsqueeze(0)
    input_intrinsics_bN33 = input_intrinsics.unsqueeze(0)

    # Create Cache3D_BufferSelector (mirrors image_input)
    cache = Cache3D_BufferSelector(
        frame_buffer_max=args.frame_buffer_max,
        input_image=input_images_bNCHW,      # [1, N, C, H, W]
        input_depth=input_depths_bN1HW,      # [1, N, 1, H, W]
        input_mask=input_masks_bN1HW,        # [1, N, 1, H, W]
        input_w2c=input_w2cs_bN44,           # [1, N, 4, 4]
        input_intrinsics=input_intrinsics_bN33,  # [1, N, 3, 3]
        filter_points_threshold=args.filter_points_threshold,
        input_format=["B", "N", "C", "H", "W"],
        foreground_masking=args.foreground_masking,
    )
    
    # Generate trajectory (same structure as image_input)
    if args.trajectory_generation_method == "action_based_movement":
        assert args.trajectory in ["left", "right", "up", "down", "zoom_in", "zoom_out", "clockwise", "counterclockwise", "none"]
        assert args.camera_rotation in ["center_facing", "no_rotation", "trajectory_aligned"]
        assert args.movement_distance is not None
        
        if args.reference_frame >= N or args.reference_frame < 0:
            raise ValueError(f"Reference frame index {args.reference_frame} is out of range. Must be between 0 and {N-1}")
        
        initial_w2c = input_w2cs[args.reference_frame]
        initial_intrinsics = input_intrinsics[args.reference_frame]
        
        try:
            generated_w2cs, generated_intrinsics = generate_camera_trajectory(
                trajectory_type=args.trajectory,
                initial_w2c=initial_w2c,
                initial_intrinsics=initial_intrinsics,
                num_frames=args.num_video_frames,
                movement_distance=args.movement_distance,
                camera_rotation=args.camera_rotation,
                center_depth=1.0,
                device=device.type,
            )
        except (ValueError, NotImplementedError) as e:
            log.critical(f"Failed to generate trajectory: {e}")
            raise
    
    elif args.trajectory_generation_method == "pixel_focusing":
        assert args.target_pixel_x is not None
        assert args.target_pixel_y is not None
        assert args.movement_ratio is not None
        assert args.start_transition_frames is not None
        assert args.end_transition_frames is not None

        if args.reference_frame >= N or args.reference_frame < 0:
            raise ValueError(f"Reference frame index {args.reference_frame} is out of range. Must be between 0 and {N-1}")

        initial_w2c = input_w2cs[args.reference_frame]
        initial_intrinsics = input_intrinsics[args.reference_frame]

        try:
            generated_w2cs, generated_intrinsics = generate_pixel_focused_trajectory(
                initial_w2c=initial_w2c,
                initial_intrinsics=initial_intrinsics,
                target_pixel=(args.target_pixel_x, args.target_pixel_y),
                num_frames=args.num_video_frames,
                movement_ratio=args.movement_ratio,
                start_transition_frames=args.start_transition_frames,
                end_transition_frames=args.end_transition_frames,
                depth_map=input_depths,  # [N, 1, H, W]
                device=device.type,
            )
        except (ValueError, NotImplementedError) as e:
            log.critical(f"Failed to generate trajectory: {e}")
            raise

    elif args.trajectory_generation_method == "source_to_target_linear_interpolation":
        assert args.source_pose_path is not None
        assert args.target_pose_path is not None
        
        # For Waymo, source/target are pose .npy paths (4x4 W2C); load them directly
        source_pose = np.load(args.source_pose_path).astype(np.float32)
        target_pose = np.load(args.target_pose_path).astype(np.float32)
        
        source_pose_tensor = torch.tensor(source_pose, device=device, dtype=torch.float32)
        target_pose_tensor = torch.tensor(target_pose, device=device, dtype=torch.float32)
        
        generated_w2cs = generate_source_to_target_trajectory(
            source_w2c = source_pose_tensor,
            target_w2c = target_pose_tensor, 
            num_frames = args.num_video_frames,
            start_transition_frames = args.start_transition_frames,
            end_transition_frames = args.end_transition_frames,
            device = device,
        )
        generated_intrinsics = input_intrinsics[0].unsqueeze(0).repeat(args.num_video_frames, 1, 1).unsqueeze(0)
    
    else:
        raise ValueError(f"Unknown trajectory generation method: {args.trajectory_generation_method}")
    
    # Render using Cache3D_BufferSelector
    rendered_warp_images, rendered_warp_masks = cache.render_cache(
        generated_w2cs,
        generated_intrinsics,
    )
    
    log.info(f"Rendered warp images shape: {rendered_warp_images.shape}")
    log.info(f"Rendered warp masks shape: {rendered_warp_masks.shape}")
    
    # Save rendered tensors
    log.info("Saving rendered tensors...")
    
    warp_images_path = os.path.join(args.rendered_tensor_dir, args.rendered_images_path)
    torch.save(rendered_warp_images, warp_images_path)
    log.info(f"Saved warp images to: {warp_images_path}")
    
    warp_masks_path = os.path.join(args.rendered_tensor_dir, args.rendered_masks_path)
    torch.save(rendered_warp_masks, warp_masks_path)
    log.info(f"Saved warp masks to: {warp_masks_path}")
    
    return


def main():
    """Main function to run the rendering creation (mirrors image_input)."""
    parser = create_parser()
    args = parser.parse_args()
    if args.prompt is None:
        args.prompt = ""
    args.disable_guardrail = True
    args.disable_prompt_upsampler = True
    
    # Handle relative paths (mirror behavior)
    if not os.path.isabs(args.input_folder):
        args.input_folder = os.path.join("..", args.input_folder)
    
    if args.source_pose_path is not None and not os.path.isabs(args.source_pose_path):
        args.source_pose_path = os.path.join("..", args.source_pose_path)
    if args.target_pose_path is not None and not os.path.isabs(args.target_pose_path):
        args.target_pose_path = os.path.join("..", args.target_pose_path)

    os.makedirs(args.rendered_tensor_dir, exist_ok=True)
    
    create_rendering(args)


if __name__ == "__main__":
    main()