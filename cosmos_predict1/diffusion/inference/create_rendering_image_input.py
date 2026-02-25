#!/usr/bin/env python3
"""
Create Rendering Script for GEN3C (Image Input)
This script takes an input image and generates rendered warp images and masks
that can be used with the diffusion_only.py script.
"""

import argparse
import os
import cv2
import torch
import numpy as np
from typing import Optional, Tuple
import torch.nn.functional as F
import json
import sys
from pathlib import Path

# Import necessary modules from the original codebase
from moge.model.v1 import MoGeModel
from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_Buffer
from cosmos_predict1.diffusion.inference.camera_utils import generate_camera_trajectory
from cosmos_predict1.diffusion.inference.camera_sequence_generation import generate_source_to_target_trajectory, load_map_to_camera_tf_matrix, generate_pixel_focused_trajectory
from cosmos_predict1.utils import log, misc
from cosmos_predict1.diffusion.inference.gen3c_single_image import _predict_moge_depth

from cosmos_predict1.diffusion.inference.inference_utils import (
    add_common_arguments,
)

# Alignment helper (for optional depth alignment to LiDAR)
from cosmos_predict1.diffusion.inference.camera_utils import _align_inv_depth_to_depth
from cosmos_predict1.diffusion.inference.create_rendering_multiview_waymo_image_input import _resize_sparse_depth

torch.enable_grad(False)

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the rendering script."""
    parser = argparse.ArgumentParser(description="Create rendered warp images and masks from input image (Image Input Version)")
    # Add common arguments
    add_common_arguments(parser)

    parser.add_argument(
        "--input_image_path",
        type=str,
        required=True,
        help="Input image path for conditioning"
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
        help="Method for generating camera trajectory: action_based_movement (default), pixel_focusing, or source_to_target_linear_interpolation"
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
        "--source_meta_path",
        type=str,
        default=None,
        help="Path to source metadata JSON file for source_to_target_linear_interpolation method"
    )
    
    parser.add_argument(
        "--target_meta_path",
        type=str,
        default=None,
        help="Path to target metadata JSON file for source_to_target_linear_interpolation method"
    )
    
    parser.add_argument(
        "--depth_estimator",
        type=str,
        choices=["moge", "depthanythingv2"],
        default="moge",
        help="Depth estimation model to use: moge (default) or depthanythingv2"
    )
    
    parser.add_argument(
        "--default_fx",
        type=float,
        default=739.75492315,
        help="Default focal length x for DepthAnythingV2 (default: 739.75492315)"
    )
    
    parser.add_argument(
        "--default_fy",
        type=float,
        default=741.66148189,
        help="Default focal length y for DepthAnythingV2 (default: 741.66148189)"
    )
    
    parser.add_argument(
        "--default_cx",
        type=float,
        default=605.94283506,
        help="Default principal point x for DepthAnythingV2 (default: 605.94283506)"
    )
    
    parser.add_argument(
        "--default_cy",
        type=float,
        default=343.51934258,
        help="Default principal point y for DepthAnythingV2 (default: 343.51934258)"
    )
    
    # LiDAR depth alignment
    parser.add_argument(
        "--align_depth_with_lidar",
        action="store_true",
        help="If set, align predicted depth to LiDAR depth using valid LiDAR mask"
    )
    
    parser.add_argument(
        "--lidar_path",
        type=str,
        default=None,
        help="Path to LiDAR depth map (.npy file) for depth alignment. Required when --align_depth_with_lidar is set."
    )
    
    return parser

def create_rendering(args) -> None:
    """
    Create rendering from input image.
    
    Args:
        args: Parsed command line arguments containing all necessary parameters
    """
    misc.set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Choose depth estimator based on argument
    if args.depth_estimator == "moge":
        moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
        
        (
            moge_image_b1chw_float,
            moge_depth_b11hw,
            moge_mask_b11hw,
            moge_initial_w2c_b144,
            moge_intrinsics_b133,
        ) = _predict_moge_depth(args.input_image_path, args.height, args.width, device, moge_model)
        
        # Use MoGe outputs
        input_image = moge_image_b1chw_float[:, 0].clone()
        input_depth = moge_depth_b11hw[:, 0]
        input_mask = moge_mask_b11hw[:, 0]
        initial_w2c = moge_initial_w2c_b144[:, 0]
        initial_intrinsics = moge_intrinsics_b133[:, 0]
        
        # Optional LiDAR alignment for MoGe
        if args.align_depth_with_lidar:
            if args.lidar_path is None:
                raise ValueError("--lidar_path is required when --align_depth_with_lidar is set")
            
            lidar_path = Path(args.lidar_path)
            if not lidar_path.exists():
                raise FileNotFoundError(f"LiDAR file not found: {lidar_path}")
            
            lidar_depth = np.load(lidar_path).astype(np.float32)
            if lidar_depth.shape != (args.height, args.width):
                # Use sparse-aware resizing for LiDAR data
                lidar_depth = _resize_sparse_depth(lidar_depth, (args.height, args.width))
            
            # Align MoGe depth with LiDAR depth
            pred_depth = input_depth[0].cpu().numpy()  # [H, W]
            lidar_t = torch.from_numpy(lidar_depth).float()
            pred_t = torch.from_numpy(pred_depth).float()
            target_mask = lidar_t > 0
            
            depth_aligned = _align_inv_depth_to_depth(
                1.0 / torch.clamp_min(pred_t, 1e-6),
                lidar_t,
                target_mask=target_mask
            )
            input_depth[0] = depth_aligned
        
    elif args.depth_estimator == "depthanythingv2":
        assert args.default_fx is not None
        assert args.default_fy is not None
        assert args.default_cx is not None
        assert args.default_cy is not None
        
        base = Path.cwd().parent.parent
        sys.path.insert(0, str(base / "Depth-Estimation" / "Depth-Anything-V2" / "metric_depth"))
        # sys.path.insert(0, str(Path(__file__).parent.parent / 'Depth-Estimation' / s'Depth-Anything-V2' / 'metric_depth'))
        from depth_anything_v2.dpt import DepthAnythingV2
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

        # Read the input image
        input_image_np = cv2.imread(args.input_image_path)
        
        # Get actual image dimensions for scaling
        actual_height, actual_width = input_image_np.shape[:2]
        
        # Convert BGR to RGB and resize
        input_image_rgb = cv2.cvtColor(input_image_np, cv2.COLOR_BGR2RGB)
        input_image_resized = cv2.resize(input_image_rgb, (args.width, args.height))
        
        # Convert to torch tensor and normalize to [-1, 1]
        input_image_tensor = torch.from_numpy(input_image_resized).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        input_image_tensor = input_image_tensor.to(device)

        # Predict depth
        with torch.no_grad():
            depth_map = dav2_model.infer_image(input_image_np)
            depth_map = cv2.resize(depth_map, (args.width, args.height))

        # Convert depth map to torch tensor and format for cache
        depth_tensor = torch.from_numpy(depth_map).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        depth_tensor = depth_tensor.to(device)
        
        # Optional LiDAR alignment for DepthAnythingV2
        if args.align_depth_with_lidar:
            if args.lidar_path is None:
                raise ValueError("--lidar_path is required when --align_depth_with_lidar is set")
            
            lidar_path = Path(args.lidar_path)
            if not lidar_path.exists():
                raise FileNotFoundError(f"LiDAR file not found: {lidar_path}")
            
            lidar_depth = np.load(lidar_path).astype(np.float32)
            if lidar_depth.shape != (args.height, args.width):
                # Use sparse-aware resizing for LiDAR data
                lidar_depth = _resize_sparse_depth(lidar_depth, (args.height, args.width))
            
            # Align DepthAnythingV2 depth with LiDAR depth
            pred_depth = depth_tensor[0, 0].cpu().numpy()  # [H, W]
            lidar_t = torch.from_numpy(lidar_depth).float()
            pred_t = torch.from_numpy(pred_depth).float()
            target_mask = lidar_t > 0
            
            depth_aligned = _align_inv_depth_to_depth(
                1.0 / torch.clamp_min(pred_t, 1e-6),
                lidar_t,
                target_mask=target_mask
            )
            depth_tensor[0, 0] = depth_aligned
        
        input_image = input_image_tensor  # [1, C, H, W]
        input_depth = depth_tensor  # [1, 1, H, W]
        input_mask = torch.ones_like(depth_tensor)  # [1, 1, H, W] 

        initial_w2c = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)  # [1, 4, 4]
        
        fx = args.default_fx * (args.width / actual_width)  
        fy = args.default_fy * (args.height / actual_height)
        cx = args.default_cx * (args.width / actual_width)
        cy = args.default_cy * (args.height / actual_height)
        
        intrinsics_matrix = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], device=device, dtype=torch.float32)
        
        initial_intrinsics = intrinsics_matrix.unsqueeze(0)

    else:
        raise ValueError(f"Unknown depth estimator: {args.depth_estimator}")
    
    frame_buffer_max = 2
    generator = torch.Generator(device=device).manual_seed(args.seed)

    cache = Cache3D_Buffer(
        frame_buffer_max=frame_buffer_max,
        generator=generator,
        noise_aug_strength=args.noise_aug_strength,
        input_image=input_image,  # [B, C, H, W]
        input_depth=input_depth,  # [B, 1, H, W]
        # input_mask=input_mask,         # [B, 1, H, W]
        input_w2c=initial_w2c,            # [B, 4, 4]
        input_intrinsics=initial_intrinsics,       # [B, 3, 3]
        filter_points_threshold=args.filter_points_threshold,
        foreground_masking=args.foreground_masking,
    )
    
    initial_cam_w2c_for_traj = initial_w2c[0]
    initial_cam_intrinsics_for_traj = initial_intrinsics[0]
    
    if args.trajectory_generation_method == "action_based_movement":
        assert args.trajectory in ["left", "right", "up", "down", "zoom_in", "zoom_out", "clockwise", "counterclockwise", "none"]
        assert args.camera_rotation in ["center_facing", "no_rotation", "trajectory_aligned"]
        assert args.movement_distance is not None
        
        try:
            generated_w2cs, generated_intrinsics = generate_camera_trajectory(
                trajectory_type=args.trajectory,
                initial_w2c=initial_cam_w2c_for_traj,
                initial_intrinsics=initial_cam_intrinsics_for_traj,
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

        try:
            generated_w2cs, generated_intrinsics = generate_pixel_focused_trajectory(
                initial_w2c=initial_cam_w2c_for_traj,
                initial_intrinsics=initial_cam_intrinsics_for_traj,
                target_pixel=(args.target_pixel_x, args.target_pixel_y),
                num_frames=args.num_video_frames,
                movement_ratio=args.movement_ratio,
                start_transition_frames=args.start_transition_frames,
                end_transition_frames=args.end_transition_frames,
                depth_map=input_depth.squeeze(1), # [B, H, W]
                device=device.type,
            )
        except (ValueError, NotImplementedError) as e:
            log.critical(f"Failed to generate trajectory: {e}")
            raise

    elif args.trajectory_generation_method == "source_to_target_linear_interpolation":
        assert args.source_meta_path is not None
        assert args.target_meta_path is not None
        
        with open(args.source_meta_path) as f:
            source_meta = json.load(f)
        with open(args.target_meta_path) as f:
            target_meta = json.load(f)
        
        # Get world-to-camera poses
        source_pose = load_map_to_camera_tf_matrix(source_meta)
        target_pose = load_map_to_camera_tf_matrix(target_meta)
        
        # Convert to torch tensors
        source_pose_tensor = torch.tensor(source_pose, device=device, dtype=torch.float32)
        target_pose_tensor = torch.tensor(target_pose, device=device, dtype=torch.float32)
        
        # Make source pose the reference (identity) and transform target pose relative to it
        # source_pose is now identity: [I]
        # target_pose becomes relative: source_pose^(-1) * target_pose
        source_pose_inv = torch.inverse(source_pose_tensor)
        target_pose_relative = torch.matmul(target_pose_tensor, source_pose_inv)
        
        # Create identity pose for source (reference)
        source_pose_identity = torch.eye(4, device=device, dtype=torch.float32)
        
        generated_w2cs = generate_source_to_target_trajectory(
            source_w2c = source_pose_identity,
            target_w2c = target_pose_relative, 
            num_frames = args.num_video_frames,
            device = device,
        )
        generated_intrinsics = initial_cam_intrinsics_for_traj.unsqueeze(0).repeat(args.num_video_frames, 1, 1).unsqueeze(0)
    
    else:
        raise ValueError(f"Unknown trajectory generation method: {args.trajectory_generation_method}")

    rendered_warp_images, rendered_warp_masks = cache.render_cache(
        generated_w2cs,
        generated_intrinsics,
    )
    
    log.info(f"Rendered warp images shape: {rendered_warp_images.shape}")
    log.info(f"Rendered warp masks shape: {rendered_warp_masks.shape}")
    
    # Save rendered tensors
    log.info("Saving rendered tensors...")
    
    # Save warp images
    warp_images_path = os.path.join(args.rendered_tensor_dir, args.rendered_images_path)
    torch.save(rendered_warp_images, warp_images_path)
    log.info(f"Saved warp images to: {warp_images_path}")
    
    # Save warp masks
    warp_masks_path = os.path.join(args.rendered_tensor_dir, args.rendered_masks_path)
    torch.save(rendered_warp_masks, warp_masks_path)
    log.info(f"Saved warp masks to: {warp_masks_path}")
    
    return

def main():
    """Main function to run the rendering creation."""
    parser = create_parser()
    args = parser.parse_args()
    if args.prompt is None:
        args.prompt = ""
    args.disable_guardrail = True
    args.disable_prompt_upsampler = True
    
    # Only prepend ".." if the input image path is relative
    # This handles the case where the script runs from GEN3C directory but input path is relative to GEN3C-Project
    if not os.path.isabs(args.input_image_path):
        args.input_image_path = os.path.join("..", args.input_image_path)
    
    # args.checkpoint_name = "Gen3C/checkpoints"

    if args.source_meta_path is not None and not os.path.isabs(args.source_meta_path):
        args.source_meta_path = os.path.join("..", args.source_meta_path)
    if args.target_meta_path is not None and not os.path.isabs(args.target_meta_path):
        args.target_meta_path = os.path.join("..", args.target_meta_path)

    if args.lidar_path is not None and not os.path.isabs(args.lidar_path):
        args.lidar_path = os.path.join("..", args.lidar_path)

    os.makedirs(args.rendered_tensor_dir, exist_ok=True)
    
    create_rendering(args)
        

if __name__ == "__main__":
    main() 