import os
import torch
import numpy as np
import cv2
import time

from sam2.build_sam import build_sam2_camera_predictor

from camera import Camera

def add_mask_overlay(frame, out_obj_ids, out_mask_logits, fps=None):
    height, width = frame.shape[:2]
    # Check mask dimensions
    mask = (out_mask_logits[0] > 0.0).cpu().numpy()    
    if mask.shape[0] == 1:
        mask = mask.squeeze(0)  # Remove the extra dimension
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    red_mask = np.zeros((height, width, 3), dtype=np.uint8)
    red_mask[mask == 1] = [0, 0, 255]  # Red color
    alpha = 0.5  # Transparency factor
    overlay = cv2.addWeighted(frame, 1, red_mask, alpha, 0)
    
    # Add FPS text if provided
    if fps is not None:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(overlay, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return overlay


device = torch.device("cuda")
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = "//home/erl-xarm6/dwait_ws/SAM2-RealTime-Webcam/checkpoints/sam2.1_hiera_small.pt"
model_cfg = "//home/erl-xarm6/dwait_ws/SAM2-RealTime-Webcam/checkpoints/sam2.1_hiera_s.yaml"

# Configuration options
SHOW_LIVE_VIEW = True  # Set to True to display segmentation results in a window
SAVE_TO_DISK = False   # Set to True to save frames to disk
WINDOW_NAME = "SAM2 Real-Time Segmentation"
OBJECT_COLOR = 'r'  # Color of object to detect: 'g' for green, 'r' for red

# Speed optimization options
USE_VOS_OPTIMIZED = False  # VOS optimization not available in this codebase - set to False
REDUCE_IMAGE_SIZE = 512   # Reduce from default 1024 to 512 for faster processing (options: 512, 768, 1024)
DISABLE_POSTPROCESSING = False  # Disable postprocessing features for speed (may reduce quality slightly)
CAMERA_RESOLUTION = (480, 360)  # Lower camera resolution for faster processing: (width, height)

# Build predictor with speed optimizations
hydra_overrides = []
if REDUCE_IMAGE_SIZE != 1024:
    hydra_overrides.append(f"++model.image_size={REDUCE_IMAGE_SIZE}")
    print(f"Using reduced image size: {REDUCE_IMAGE_SIZE}x{REDUCE_IMAGE_SIZE} for faster inference")

predictor = build_sam2_camera_predictor(
    model_cfg, 
    sam2_checkpoint,
    vos_optimized=USE_VOS_OPTIMIZED,
    apply_postprocessing=not DISABLE_POSTPROCESSING,
    hydra_overrides_extra=hydra_overrides
)

# Initialize RealSense camera
print("Initializing RealSense camera...")
camera = Camera(debug=False, imsize=CAMERA_RESOLUTION)
print(f"Camera initialized successfully with resolution: {CAMERA_RESOLUTION[0]}x{CAMERA_RESOLUTION[1]}")

# Wait for first frame from camera with object detection
print("Waiting for first frame and detecting object...")
frame = None
object_pos = None
max_attempts = 100  # Maximum attempts to detect object
attempt = 0

while frame is None or object_pos is None:
    result = camera.fetch_image(detect_object=True, color=OBJECT_COLOR)  # Detect object
    if result is not None:
        frame, _, pos = result  # Get color image, depth, and position
        if pos is not None:
            object_pos = pos
            print(f"First frame captured and object detected at: {object_pos}")
            break
        else:
            attempt += 1
            if attempt >= max_attempts:
                print(f"Warning: object not detected after {max_attempts} attempts. Using center of image.")
                # Fallback to center if object not detected
                height, width = frame.shape[:2]
                object_pos = np.array([width // 2, height // 2])
                break
            print(f"object not detected, attempt {attempt}/{max_attempts}...")
    time.sleep(0.1)

# read the first frame
predictor.load_first_frame(frame)

ann_frame_idx = 0  # the frame index we interact with

ann_obj_id = (
    1  # give a unique id to each object we interact with (it can be any integers)
)

# Use detected object position as the initial point
points = np.array([[object_pos[0], object_pos[1]]], dtype=np.float32)
print(f"Using detected object position {object_pos} as initial segmentation point")

# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], dtype=np.int32)

_, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# out_obj_ids, out_mask_logits = predictor.track(frame)
overlay = add_mask_overlay(frame, out_obj_ids, out_mask_logits, fps=None)

# Show live view if enabled
if SHOW_LIVE_VIEW:
    cv2.imshow(WINDOW_NAME, overlay)
    print(f"Displaying initial frame with mask. Press 'q' to quit.")
    
# Save to disk if enabled
if SAVE_TO_DISK:
    os.makedirs("temp_data/input_images", exist_ok=True)
    output_path = os.path.join("temp_data", "input_images", f"result_frame_{ann_frame_idx:04d}.jpg")
    cv2.imwrite(output_path, overlay)
    print(f"Saved initial frame with mask: {output_path}")

# start tracking videos
print("Starting continuous tracking...")
if SHOW_LIVE_VIEW:
    print(f"Live view enabled. Press 'q' in the window to quit.")
if SAVE_TO_DISK:
    print(f"Saving frames to disk: temp_data/input_images/")

# FPS tracking variables
fps = 0.0
fps_update_time = time.time()
fps_frame_count = 0
fps_window = 30  # Number of frames to average for FPS calculation
frame_times = []

ann_frame_idx = 1
while True:
    try:
        # Start timing for FPS calculation
        frame_start_time = time.time()
        
        # Get frame from camera
        result = camera.fetch_image()
        if result is not None:
            frame, _ = result  # Get color image
            # Track objects in the new frame
            out_obj_ids, out_mask_logits = predictor.track(frame)
            
            # Calculate FPS
            frame_time = time.time() - frame_start_time
            frame_times.append(frame_time)
            if len(frame_times) > fps_window:
                frame_times.pop(0)
            
            # Update FPS display periodically
            fps_frame_count += 1
            if fps_frame_count >= fps_window or time.time() - fps_update_time > 1.0:
                if len(frame_times) > 0:
                    avg_frame_time = np.mean(frame_times)
                    fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
                fps_update_time = time.time()
            
            overlay = add_mask_overlay(frame, out_obj_ids, out_mask_logits, fps)
            
            # Show live view if enabled
            if SHOW_LIVE_VIEW:
                cv2.imshow(WINDOW_NAME, overlay)
                # Check for 'q' key press to quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n'q' pressed - stopping tracking...")
                    break
            
            # Save to disk if enabled
            if SAVE_TO_DISK:
                output_path = os.path.join("temp_data", "input_images", f"result_frame_{ann_frame_idx:04d}.jpg")
                cv2.imwrite(output_path, overlay)
            
            ann_frame_idx += 1
        else:
            print("Warning: No frame available from camera, retrying...")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt - stopping tracking...")
        break
    except Exception as e:
        print(f"Error during tracking: {e}")
        break

# Clean up
if SHOW_LIVE_VIEW:
    cv2.destroyAllWindows()
print("Closing camera...")
camera.close()
print("Camera closed successfully")





