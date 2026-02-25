#!/usr/bin/env python3
"""
Simplified Diffusion-Only Version of GEN3C
This version focuses only on the diffusion model without rendering components.
It generates a single video sequence with num_ar_iterations=1.
"""

import argparse
import os
import cv2
import torch
import numpy as np
from typing import Optional, Tuple
import torch.nn.functional as F

# Import necessary modules from the original codebase
from cosmos_predict1.diffusion.inference.gen3c_pipeline import Gen3cPipeline
from cosmos_predict1.utils import log, misc
from cosmos_predict1.utils.io import save_video
from cosmos_predict1.diffusion.inference.vipe_utils import load_vipe_data

from cosmos_predict1.diffusion.inference.inference_utils import (
    add_common_arguments,
)

torch.enable_grad(False)

def validate_args(args):
    assert args.num_video_frames is not None, "num_video_frames must be provided"
    assert args.num_video_frames == 121, "num_video_frames must be 121"
    
    # Validate that either input_image_path or vipe_path is provided
    if args.vipe_path is None and args.input_image_path is None:
        raise ValueError("Either --input_image_path or --vipe_path must be provided")
    
    if args.vipe_path is not None and args.input_image_path is not None:
        log.warning("Both --vipe_path and --input_image_path provided. Using --vipe_path.")

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the simplified diffusion-only version."""
    parser = argparse.ArgumentParser(description="Diffusion-only video generation script")
    
    add_common_arguments(parser)
    
    # 🔧 Override ONLY the default of --video_save_folder
    for action in parser._actions:
        if "--video_save_folder" in action.option_strings:
            action.default = "test_results/output/"
            # Optional but nice: keep argparse’s internal defaults in sync
            parser.set_defaults(video_save_folder="test_results/output/")
            break
    
    parser.add_argument(
        "--prompt_upsampler_dir",
        type=str,
        default="Pixtral-12B",
        help="Prompt upsampler weights directory relative to checkpoint_dir",
    )
    parser.add_argument(
        "--input_image_path",
        type=str,
        required=False,
        help="Input image path for conditioning (required when not using ViPE)"
    )
    
    parser.add_argument(
        "--vipe_path",
        type=str,
        required=False,
        help="Path to ViPE clip root directory or mp4 file under rgb/ (alternative to input_image_path)"
    )
    
    parser.add_argument(
        "--vipe_starting_frame_idx",
        type=int,
        default=0,
        help="Starting frame index within the ViPE video (default: 0)"
    )
    
    parser.add_argument(
        "--rendered_tensor_dir",
        type=str,
        default="../test_results/rendered_tensor_dir",
        help="Directory containing the rendered tensor files"
    )
    
    parser.add_argument(
        "--rendered_images_path",
        type=str,
        default="rendered_warp_images.pt",
        help="Filename for the rendered warp images tensor"
    )
    
    parser.add_argument(
        "--rendered_masks_path",
        type=str,
        default="rendered_warp_masks.pt",
        help="Filename for the rendered warp masks tensor"
    )

    parser.add_argument(
        "--save_buffer",
        action="store_true",
        help="Whether to save the warped images buffer (True/False, default: False)"
    )
    
    parser.add_argument(
        "--save_output_video_numpy",
        action="store_true",
        help="Whether to save the output video as a numpy array (True/False, default: False)"
    )
    
    return parser

def load_tensor_from_path(tensor_path: str, device: torch.device) -> torch.Tensor:
    """Load a tensor from a file path."""
    if not os.path.exists(tensor_path):
        raise FileNotFoundError(f"Tensor file not found: {tensor_path}")
    
    tensor = torch.load(tensor_path, map_location=device)
    log.info(f"Loaded tensor from {tensor_path} with shape: {tensor.shape}")
    return tensor


def generate_video_diffusion_only(args) -> None:
    """Generate video using only the diffusion model without rendering."""
    misc.set_random_seed(args.seed)
    inference_type = "video2world"
    validate_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the diffusion pipeline
    pipeline = Gen3cPipeline(
        inference_type=inference_type,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name="Gen3C-Cosmos-7B",
        prompt_upsampler_dir=args.prompt_upsampler_dir,
        enable_prompt_upsampler=not args.disable_prompt_upsampler,
        offload_network=args.offload_diffusion_transformer,
        offload_tokenizer=args.offload_tokenizer,
        offload_text_encoder_model=args.offload_text_encoder_model,
        offload_prompt_upsampler=args.offload_prompt_upsampler,
        offload_guardrail_models=args.offload_guardrail_models,
        disable_guardrail=args.disable_guardrail,
        disable_prompt_encoder=args.disable_prompt_encoder,
        guidance=args.guidance,
        num_steps=args.num_steps,
        height=args.height,
        width=args.width,
        fps=args.fps,
        num_video_frames=121,
        seed=args.seed,
    )
    
    # Load the pre-rendered warp images and masks
    log.info("Loading pre-rendered warp images and masks...")
    
    # Construct full paths using directory and filenames
    rendered_images_full_path = os.path.join(args.rendered_tensor_dir, args.rendered_images_path)
    rendered_masks_full_path = os.path.join(args.rendered_tensor_dir, args.rendered_masks_path)
    
    rendered_warp_images = load_tensor_from_path(rendered_images_full_path, device)
    # assert rendered_warp_images.shape == (1, 121, 1, 3, args.height, args.width), f"rendered_warp_images must have shape [1, 121, 1, 3, {args.height}, {args.width}], got {rendered_warp_images.shape}"
    
    rendered_warp_masks = load_tensor_from_path(rendered_masks_full_path, device)
    # assert rendered_warp_masks.shape == (1, 121, 1, 1, args.height, args.width), f"rendered_warp_masks must have shape [1, 121, 1, 1, {args.height}, {args.width}], got {rendered_warp_masks.shape}"
    
    log.info(f"Warp images shape: {rendered_warp_images.shape}")
    log.info(f"Warp masks shape: {rendered_warp_masks.shape}")
    
    # Ensure tensors are on the correct device
    rendered_warp_images = rendered_warp_images.to(device)
    rendered_warp_masks = rendered_warp_masks.to(device)
    
    # Initialize buffer collection if save_buffer is enabled
    all_rendered_warps = []
    if args.save_buffer:
        all_rendered_warps.append(rendered_warp_images.clone().cpu())
    
    # Handle ViPE data loading if vipe_path is provided
    if args.vipe_path is not None:
        log.info(f"Loading ViPE data from: {args.vipe_path}")
        try:
            (
                image_bchw_float,
                depth_b1hw,
                mask_b1hw,
                initial_w2c_b44,
                intrinsics_b33,
            ) = load_vipe_data(
                vipe_root_or_mp4=args.vipe_path,
                starting_frame_idx=args.vipe_starting_frame_idx,
                resize_hw=(args.height, args.width),
                crop_hw=(args.height, args.width),
                num_frames=args.num_video_frames,
            )
            
            # Use the first frame from ViPE data as input image
            input_image = image_bchw_float[0].unsqueeze(0).unsqueeze(2).to(device)  # [1, C, 1, H, W]
            log.info(f"Successfully loaded ViPE data. Using frame 0 as input image with shape: {input_image.shape}")
            
        except Exception as e:
            log.critical(f"Failed to load ViPE data: {e}")
            return
    else:
        # Use traditional input image path
        input_image = args.input_image_path
    
    generated_output = pipeline.generate(
        prompt=args.prompt,
        image_path=input_image,
        negative_prompt=args.negative_prompt,
        rendered_warp_images=rendered_warp_images,
        rendered_warp_masks=rendered_warp_masks,
    )
    
    if generated_output is None:
        log.critical("Video generation failed!")
        return
    
    video, final_prompt = generated_output
    log.info(f"Video generation completed successfully!")
    log.info(f"Generated video shape: {video.shape}")
    
    
    
    # Process buffer video if save_buffer is enabled
    final_video_to_save = video
    final_width = args.width
    
    if args.save_buffer and all_rendered_warps:
        squeezed_warps = [t.squeeze(0) for t in all_rendered_warps] # Each is (T_chunk, n_i, C, H, W)

        if squeezed_warps:
            n_max = max(t.shape[1] for t in squeezed_warps)

            padded_t_list = []
            for sq_t in squeezed_warps:
                # sq_t shape: (T_chunk, n_i, C, H, W)
                current_n_i = sq_t.shape[1]
                padding_needed_dim1 = n_max - current_n_i

                pad_spec = (0,0, # W
                            0,0, # H
                            0,0, # C
                            0,padding_needed_dim1, # n_i
                            0,0) # T_chunk
                padded_t = F.pad(sq_t, pad_spec, mode='constant', value=-1.0)
                padded_t_list.append(padded_t)

            full_rendered_warp_tensor = torch.cat(padded_t_list, dim=0)

            T_total, _, C_dim, H_dim, W_dim = full_rendered_warp_tensor.shape
            buffer_video_TCHnW = full_rendered_warp_tensor.permute(0, 2, 3, 1, 4)
            buffer_video_TCHWstacked = buffer_video_TCHnW.contiguous().view(T_total, C_dim, H_dim, n_max * W_dim)
            buffer_video_TCHWstacked = (buffer_video_TCHWstacked * 0.5 + 0.5) * 255.0
            buffer_numpy_TCHWstacked = buffer_video_TCHWstacked.cpu().numpy().astype(np.uint8)
            buffer_numpy_THWC = np.transpose(buffer_numpy_TCHWstacked, (0, 2, 3, 1))

            final_video_to_save = np.concatenate([buffer_numpy_THWC, final_video_to_save], axis=2)
            final_width = args.width * (1 + n_max)
            log.info(f"Concatenating video with {n_max} warp buffers. Final video width will be {final_width}")
        else:
            log.info("No warp buffers to save.")

    
        video_save_path = os.path.join(
            args.video_save_folder,
            f"{args.video_save_name}_with_buffer.mp4"
        )
        
        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
        
        # Save video
        save_video(
            video=final_video_to_save,
            fps=args.fps,
            H=args.height,
            W=final_width,
            video_save_quality=5,
            video_save_path=video_save_path, 
        )
        
        log.info(f"Saved video with buffer to {video_save_path}")

    final_video_to_save = video
    final_width = args.width

    video_save_path = os.path.join(
        args.video_save_folder,
        f"{args.video_save_name}.mp4"
    )

    os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
    
    # Save video
    save_video(
        video=final_video_to_save,
        fps=args.fps,
        H=args.height,
        W=final_width,
        video_save_quality=5,
        video_save_path=video_save_path, 
    )
    
    if args.save_output_video_numpy:
        numpy_save_path = os.path.join(
            args.video_save_folder,
            f"{args.video_save_name}.npy"
        )
        np.save(numpy_save_path, final_video_to_save)
        log.info(f"Saved output video numpy array to {numpy_save_path}")

def main():
    """Main function to run the diffusion-only video generation."""
    parser = create_parser()
    args = parser.parse_args()
    if args.prompt is None:
        args.prompt = ""
    args.disable_guardrail = True
    args.disable_prompt_upsampler = True
    # args.checkpoint_name = "Gen3C/checkpoints"

    # Only prepend ".." if the input image path is relative and provided
    # This handles the case where the script runs from GEN3C directory but input path is relative to GEN3C-Project
    if args.input_image_path is not None and not os.path.isabs(args.input_image_path):
        args.input_image_path = os.path.join("..", args.input_image_path)

    if not os.path.isabs(args.video_save_folder):
        args.video_save_folder = os.path.join("..", args.video_save_folder)
    
    # Handle ViPE path if provided
    if args.vipe_path is not None and not os.path.isabs(args.vipe_path):
        args.vipe_path = os.path.join("..", args.vipe_path)
    
    assert os.path.exists(args.rendered_tensor_dir), f"Rendered tensor directory does not exist: {args.rendered_tensor_dir}"
    
    generate_video_diffusion_only(args)

if __name__ == "__main__":
    main() 