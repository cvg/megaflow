"""
demo_track.py - MegaFlow point tracking demo.

Tracks points across video frames using MegaFlow's flow-based tracking.
Auto-downloads pretrained models from HuggingFace.
Dynamically resizes inputs based on a fixed width to match optimal model scale,
and restores the tracking visualizations to the original resolution.

Usage:
    python demo_track.py --input assets/example.mp4 --output output/ --fix_width 518
"""

import argparse
import os
import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from megaflow.model import MegaFlow
from megaflow.utils.basic import gridcloud2d
from megaflow.utils.visualizer import Visualizer
import megaflow.utils.frame_utils as frame_utils


def calculate_dynamic_size(orig_h, orig_w, fix_width, patch_size=14):
    """
    Calculates new dimensions by fixing the width and preserving the aspect ratio.
    Ensures the height is divisible by the patch_size constraint.
    """
    new_w = fix_width
    new_h = round(orig_h * (new_w / orig_w) / patch_size) * patch_size
        
    return int(new_h), int(new_w)


def get_frames(input_path: str, fix_width=None, patch_size=14):
    """Generator that yields (RGB frame, original_shape_tuples, native_fps).
    If fix_width is provided, resizes the frame preserving aspect ratio before yielding.
    Returns: (frame_array, (original_H, original_W), native_fps)
    """
    if os.path.isdir(input_path):
        exts = ('*.png', '*.jpg', '*.jpeg', '*.BMP', '*.bmp')
        image_paths = []
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(input_path, ext)))
        image_paths = sorted(image_paths)
        if not image_paths:
            raise FileNotFoundError(f"No images found in {input_path}")
        
        for path in image_paths:
            img = frame_utils.read_gen(path)
            img = np.array(img).astype(np.uint8)[..., :3]
            orig_shape = img.shape[:2]
            
            if fix_width is not None:
                new_h, new_w = calculate_dynamic_size(orig_shape[0], orig_shape[1], fix_width, patch_size)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
            yield img, orig_shape, None
    else:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")
            
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if native_fps <= 0 or np.isnan(native_fps):
            native_fps = None
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            orig_shape = frame.shape[:2]
            
            if fix_width is not None:
                new_h, new_w = calculate_dynamic_size(orig_shape[0], orig_shape[1], fix_width, patch_size)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
            yield frame, orig_shape, native_fps
        cap.release()


def process_sequence(args, model, device):
    print(f"Loading frames from {args.input}...")
    
    native_size = None
    input_video_fps = None
    input_image = []
    
    for frame, orig_shape, extracted_fps in get_frames(args.input, args.fix_width):
        if native_size is None:
            native_size = orig_shape  

        # Capture the FPS from the first yielded frame
        if input_video_fps is None and extracted_fps is not None:
            input_video_fps = extracted_fps
            
        img_t = torch.from_numpy(frame).permute(2, 0, 1).float()
        input_image.append(img_t)
        
    if len(input_image) < 2:
        print("Sequence too short (requires at least 2 frames).")
        return

    frames = torch.stack(input_image, dim=0)[None].to(device)
    B, T, _, H, W = frames.shape

    grid_xy = gridcloud2d(1, H, W, norm=False, device=device).float()
    grid_xy = grid_xy.permute(0, 2, 1).reshape(1, 1, 2, H, W)  

    if device == "cuda":
        torch.cuda.empty_cache()

    compute_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast(device_type=device, dtype=compute_dtype, enabled=(device == "cuda")):
        results = model.forward_track(frames, num_reg_refine=args.iters)
        flows_e = results["flow_final"]
        traj_maps = flows_e.to(device) + grid_xy  

    rate = args.grid_size
    traj_sub = traj_maps[..., ::rate, ::rate]  

    if args.mask_path is not None:
        from PIL import Image
        mask_pil = Image.open(args.mask_path).convert("L")
        
        # Resize mask to perfectly align with the dynamically resized frames
        mask_pil = mask_pil.resize((W, H), Image.Resampling.NEAREST)
        
        mask_np = np.array(mask_pil)
        mask_np = (mask_np > 127).astype(np.uint8) * 255
        kernel = np.ones((9, 9), np.uint8)
        mask_np = cv2.erode(mask_np, kernel, iterations=1)
        segm_mask = torch.from_numpy(mask_np.copy())[None, None].to(device)
        mask_sub = segm_mask[..., ::rate, ::rate]
        pred_tracks = traj_sub.flatten(3)
        mask_flat = (mask_sub > 0).flatten()
        pred_tracks = pred_tracks[..., mask_flat]
    else:
        pred_tracks = traj_sub.flatten(3)

    pred_tracks = pred_tracks.permute(0, 1, 3, 2)  
    
    # Restore the coordinates and the frames back to their native size
    if args.restore_size and native_size is not None:
        orig_H, orig_W = native_size
        scale_x = orig_W / W
        scale_y = orig_H / H
        
        # Scale the tracking coordinates
        pred_tracks[..., 0] *= scale_x
        pred_tracks[..., 1] *= scale_y
        
        # Upscale the frames for visualization (B, T, C, orig_H, orig_W)
        frames = F.interpolate(frames[0], size=(orig_H, orig_W), mode='bilinear', align_corners=True).unsqueeze(0)

    os.makedirs(args.output, exist_ok=True)
    vis = Visualizer(
        save_dir=args.output,
        pad_value=0,
        linewidth=1,
        tracks_leave_trace=0,
        fps=input_video_fps if input_video_fps is not None else args.fps
    )
    vis.visualize(
        frames,
        pred_tracks,
        filename=os.path.splitext(os.path.basename(args.input))[0],
        opacity=0.5,
    )
    print(f"Tracking visualization saved to: {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MegaFlow Point Tracking Demo")
    parser.add_argument("--input", type=str, required=True, help="Path to input video or image folder")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--fix_width", type=int, default=518, help="Target width for resizing, preserving aspect ratio (default: 518)")
    parser.add_argument("--restore_size", action="store_true", default=True, help="Restore upscaled frames and tracks to original dimensions before saving")
    parser.add_argument("--mask_path", type=str, default=None, help="Path to segmentation mask (optional)")
    parser.add_argument("--grid_size", type=int, default=8, help="Subsampling rate for tracked points")
    parser.add_argument("--fps", type=float, default=24.0, help="Fallback framerate for image folders (default: 24.0)")
    parser.add_argument("--iters", type=int, default=8, help="Number of refinement iterations")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading predefined model 'megaflow-track' from HuggingFace...")
    model = MegaFlow.from_pretrained("megaflow-track", device=device)
    model.eval()

    with torch.inference_mode():
        process_sequence(args, model, device)


if __name__ == "__main__":
    main()