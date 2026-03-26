"""
demo_flow.py - MegaFlow inference demo for videos or image folders.

Estimates optical flow between consecutive frames in a video or an image folder
and saves colour-coded visualizations as a video or sequence of images.
Optionally resizes frames, processes in sliding windows for long sequences,
and saves raw .flo files.

Usage:
    python demo_flow.py --input path/to/video.mp4 --output output.mp4
    python demo_flow.py --input path/to/images/ --output path/to/output_folder/ --save_flo --fix_width 952
"""

import argparse
import os
import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import imageio

from megaflow.model import MegaFlow
from megaflow.utils.flow_viz import flow_to_image
import megaflow.utils.frame_utils as frame_utils

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x


def calculate_dynamic_size(orig_h, orig_w, target_fix_width, patch_size=14):
    """
    Calculates new dimensions preserving the original aspect ratio.
    Aligns the longest edge to the target_fix_width and ensures divisibility.
    """
    if orig_w >= orig_h:
        new_w = target_fix_width
        new_h = round(orig_h * (new_w / orig_w) / patch_size) * patch_size
    else:
        new_h = target_fix_width
        new_w = round(orig_w * (new_h / orig_h) / patch_size) * patch_size
        
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
                
            # Folders do not have a native framerate
            yield img, orig_shape, None
    else:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")
            
        # Extract the native framerate directly from the video metadata
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
        
    input_scene = torch.stack(input_image, dim=0)[None]
    B, T, C, H, W = input_scene.shape
    
    is_video_out = False
    video_writer = None
    out_dir = None
    
    if args.output.lower().endswith(('.mp4', '.avi', '.mkv', '.webm')):
        is_video_out = True
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        
        # Prioritize native video FPS, fallback to args.fps for image folders
        final_fps = input_video_fps if input_video_fps is not None else args.fps
        print(f"Encoding output video at {final_fps:.2f} FPS")
        
        video_writer = imageio.get_writer(args.output, fps=final_fps, codec='libx264', macro_block_size=None)
    else:
        out_dir = args.output
        os.makedirs(out_dir, exist_ok=True)

    print(f"Processing sequence of {T} frames via window size {args.window_size}...")
    infer_window = args.window_size
    
    for start in tqdm(range(0, T - 1, infer_window - 1)):
        end = min(start + infer_window, T)
        chunk = input_scene[:, start:end].to(device)
        
        compute_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type=device, dtype=compute_dtype, enabled=(device == "cuda")):
            results_dict = model(chunk, num_reg_refine=args.iters)
            
        flow_pr = results_dict['flow_preds'][-1]
        
        if native_size is not None and getattr(args, 'restore_size', False):
            scaled_flow = F.interpolate(flow_pr.view(-1, 2, H, W), size=native_size, mode='bilinear', align_corners=True)
            
            scale_y = native_size[0] / H
            scale_x = native_size[1] / W
            scaled_flow[:, 0, :, :] *= scale_x
            scaled_flow[:, 1, :, :] *= scale_y
            
            flow_pr = scaled_flow.view(*flow_pr.shape[:2], 2, *native_size)
            
        flow_all = flow_pr[0].permute(0, 2, 3, 1).cpu().numpy()  
        
        for t, flow in enumerate(flow_all):
            frame_idx = start + t + 1  
            
            if is_video_out:
                flow_vis_rgb = flow_to_image(flow, convert_to_bgr=False)
                video_writer.append_data(flow_vis_rgb)
            else:
                flow_vis_bgr = flow_to_image(flow, convert_to_bgr=True)
                vis_file = os.path.join(out_dir, "frame%04d.png" % frame_idx)
                cv2.imwrite(vis_file, flow_vis_bgr)
                
                if args.save_flo:
                    output_file = os.path.join(out_dir, "frame%04d.flo" % frame_idx)
                    frame_utils.writeFlow(output_file, flow)
                    
    if is_video_out and video_writer is not None:
        video_writer.close()
        print(f"Saved flow video to {args.output}")
    else:
        print(f"Saved flow images (and optionally .flo files) to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MegaFlow: optical flow demo")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file or image folder")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to output video file (.mp4) or directory")
    parser.add_argument("--fix_width", type=int, default=952, help="Target size for the longest edge, preserving aspect ratio (default: 952)")
    parser.add_argument("--restore_size", action="store_true", default=True, help="Restore upscaled flow to the original dimensions before saving")
    parser.add_argument("--window_size", type=int, default=4, help="Sliding window size (infer_window) for processing")
    parser.add_argument("--save_flo", action="store_true", help="Save .flo format alongside visualizations (for folder output)")
    parser.add_argument("--fps", type=float, default=24.0, help="Fallback framerate for image folders (default: 24.0)")
    parser.add_argument("--iters", type=int, default=8, help="Number of refinement iterations (default: 8, per snippet)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input path not found: {args.input}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading predefined model 'megaflow-flow' from HuggingFace...")
    model = MegaFlow.from_pretrained("megaflow-flow", device=device)
    model.eval()

    with torch.inference_mode():
        process_sequence(args, model, device)


if __name__ == "__main__":
    main()