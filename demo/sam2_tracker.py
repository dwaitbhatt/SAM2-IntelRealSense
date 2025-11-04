"""
High-level wrapper class for SAM2 object tracking in video/camera streams.
"""
import os
import sys
import torch
import numpy as np
import cv2

from sam2.build_sam import build_sam2_camera_predictor

class SAM2Tracker:
    """
    High-level wrapper for SAM2 object tracking.
    
    Supports tracking 1 or 2 objects in a video/camera stream with automatic initialization.
    """
    
    def __init__(self, camera, model_type="small", init_pos_object_color=None, object_count=1, 
                 image_size=512, debug=False, window_name="SAM2 Tracker"):
        """
        Initialize SAM2 tracker.
        
        Args:
            camera: Camera object (must have fetch_image() method that returns (color_image, depth_image) or None).
                   Required if init_pos_object_color contains only colors (for auto-detection).
                   Can be None if frames and positions are always provided.
            model_type (str): Model type, either "tiny" or "small"
            init_pos_object_color (list): List of entries for each object. Two formats supported:
                                         - With positions: [[x1, y1, 'g']] or [[x1, y1, 'g'], [x2, y2, 'r']]
                                         - Colors only (auto-detect): [['g']] or [['g'], ['r']]
                                         If colors only, camera must be provided for auto-detection.
            object_count (int): Number of objects to track (1 or 2)
            image_size (int): Internal image size for processing (512, 768, or 1024)
            debug (bool): If True, display segmentation masks overlaid on the original image using cv2.imshow()
            window_name (str): Name of the debug display window
        """
        self.camera = camera
        self.model_type = model_type
        self.object_count = object_count
        self.image_size = image_size
        self.debug = debug
        self.window_name = window_name
        
        # Colors for different objects in overlay (BGR format)
        self.mask_colors = [
            (0, 0, 255),   # Red for object 1
            (0, 255, 0),   # Green for object 2
        ]
        
        # Validate inputs
        if object_count not in [1, 2]:
            raise ValueError(f"object_count must be 1 or 2, got {object_count}")
        
        if model_type not in ["tiny", "small"]:
            raise ValueError(f"model_type must be 'tiny' or 'small', got {model_type}")
        
        if init_pos_object_color is None:
            raise ValueError("init_pos_object_color must be provided")
        
        if len(init_pos_object_color) != object_count:
            raise ValueError(f"init_pos_object_color must have {object_count} entries, got {len(init_pos_object_color)}")
        
        # Check if colors-only mode is used (only color provided, no positions)
        self._colors_only_mode = False
        for item in init_pos_object_color:
            if len(item) == 1:
                # Only color provided
                self._colors_only_mode = True
                break
        
        # If colors-only mode, camera must be provided
        if self._colors_only_mode and camera is None:
            raise ValueError("camera must be provided when using colors-only mode (auto-detection)")
        
        # Setup CUDA optimizations
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        
        # Initialize model
        self._init_model()
        
        # Track initialization state
        self._is_initialized = False
        self._init_pos_object_color = init_pos_object_color
        
        # Object IDs (using 1-indexed IDs)
        self._obj_ids = list(range(1, object_count + 1))
        
    def _init_model(self):
        """Initialize the SAM2 model based on model_type."""
        base_path = "//home/erl-xarm6/dwait_ws/SAM2-RealTime-Webcam"
        
        if self.model_type == "tiny":
            checkpoint = f"{base_path}/checkpoints/sam2.1_hiera_tiny.pt"
            config = f"{base_path}/checkpoints/sam2.1_hiera_t.yaml"
        else:  # small
            checkpoint = f"{base_path}/checkpoints/sam2.1_hiera_small.pt"
            config = f"{base_path}/checkpoints/sam2.1_hiera_s.yaml"
        
        # Build predictor with optimizations
        hydra_overrides = []
        if self.image_size != 1024:
            hydra_overrides.append(f"++model.image_size={self.image_size}")
        
        print(f"Initializing SAM2 {self.model_type} model with image_size={self.image_size}...")
        self.predictor = build_sam2_camera_predictor(
            config,
            checkpoint,
            vos_optimized=False,
            apply_postprocessing=True,
            hydra_overrides_extra=hydra_overrides
        )
        print("Model initialized successfully")
    
    def _detect_object_positions(self, max_attempts=100):
        """
        Automatically detect object positions using camera based on colors.
        
        Args:
            max_attempts (int): Maximum attempts to detect each object
            
        Returns:
            list: List of [x, y, color] for each object
        """
        detected_positions = []
        
        for i, item in enumerate(self._init_pos_object_color):
            if len(item) == 1:
                # Only color provided - need to detect
                color = item[0]
                print(f"Auto-detecting object {i+1} with color '{color}'...")
                
                for attempt in range(max_attempts):
                    result = self.camera.fetch_image(detect_object=True, color=color)
                    if result is not None:
                        _, _, pos = result
                        if pos is not None:
                            detected_positions.append([pos[0], pos[1], color])
                            print(f"Object {i+1} detected at: [{pos[0]:.1f}, {pos[1]:.1f}]")
                            break
                    
                    if attempt == max_attempts - 1:
                        # Fallback to center if not detected
                        result = self.camera.fetch_image()
                        if result is not None:
                            frame, _ = result
                            height, width = frame.shape[:2]
                            detected_positions.append([width // 2, height // 2, color])
                            print(f"Object {i+1} not detected after {max_attempts} attempts. Using center.")
                        else:
                            raise RuntimeError(f"Could not get frame from camera for object {i+1}")
            else:
                # Position already provided
                detected_positions.append(item)
        
        return detected_positions
    
    def _initialize_tracking(self, frame):
        """
        Initialize tracking with the first frame and object positions.
        
        Args:
            frame: First frame from camera
        """
        # If colors-only mode, detect positions first
        if self._colors_only_mode:
            # Detect all object positions using camera
            init_positions = self._detect_object_positions()
        else:
            # Positions already provided
            init_positions = self._init_pos_object_color
        
        # Load first frame into predictor
        self.predictor.load_first_frame(frame)
        
        # Add prompts for each object
        ann_frame_idx = 0
        for i, item in enumerate(init_positions):
            obj_id = self._obj_ids[i]
            
            # Extract x, y from [x, y, color] format
            x, y = float(item[0]), float(item[1])
            
            # Convert position to float32 array
            points = np.array([[x, y]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)  # Positive click
            
            # Add prompt for this object
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                frame_idx=ann_frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
            )
        
        self._is_initialized = True
        print(f"Tracking initialized for {self.object_count} object(s)")
    
    def _create_overlay(self, frame, masks):
        """
        Create an overlay image with segmentation masks on top of the original frame.
        
        Args:
            frame: Original color image (BGR format)
            masks: List of segmentation masks (one per object)
            
        Returns:
            numpy.ndarray: Overlay image with masks displayed
        """
        overlay = frame.copy()
        
        for i, mask in enumerate(masks):
            if mask is None:
                continue
                
            # Create colored mask
            color = self.mask_colors[i % len(self.mask_colors)]
            colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            colored_mask[mask == 1] = color
            
            # Blend with original image
            alpha = 0.5  # Transparency factor
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, alpha, 0)
        
        return overlay
    
    def detect_in_next_frame(self, frame=None):
        """
        Detect and track objects in the next frame.        
        Automatically handles first frame initialization if called for the first time.
        
        Args:
            frame (numpy.ndarray, optional): Color image frame. If None, fetches from camera.
                                            Shape should be (H, W, 3) in BGR format.
        
        Returns:
            list: List of segmentation masks, one per object. Each mask is a numpy array
                  of shape (H, W) with values 0 or 1, where 1 indicates the object.
                  Returns None if frame cannot be obtained.
        """
        # Get frame from camera if not provided
        if frame is None:
            if self.camera is None:
                raise ValueError("frame must be provided if camera is None")
            
            result = self.camera.fetch_image()
            
            if result is None:
                return None
            
            frame, _ = result  # Get color image (ignore depth)
        
        # Validate frame
        if frame is None or not isinstance(frame, np.ndarray):
            return None
        
        # Handle first frame initialization
        if not self._is_initialized:
            # If colors-only mode, we need to use the provided frame (or get one)
            # for position detection, then use that same frame for initialization
            if self._colors_only_mode:
                # Use the provided frame or fetch one
                if frame is None:
                    result = self.camera.fetch_image()
                    if result is None:
                        return None
                    frame, _ = result
            
            self._initialize_tracking(frame)
            # After initialization, track() will use the initialized state
            # Note: First frame masks come from add_new_prompt, but we call track()
            # to properly start the tracking pipeline for subsequent frames
            out_obj_ids, out_mask_logits = self.predictor.track(frame)
        else:
            # Track objects in subsequent frames
            out_obj_ids, out_mask_logits = self.predictor.track(frame)
        
        # Extract masks for each object
        masks = []
        height, width = frame.shape[:2]
        
        # out_mask_logits is a tensor with shape (num_objects, 1, H, W) or (num_objects, H, W)
        # We need to extract one mask per object
        for i in range(self.object_count):
            mask_logits = out_mask_logits[i]
            
            # Convert to numpy if tensor
            if isinstance(mask_logits, torch.Tensor):
                mask = (mask_logits > 0.0).cpu().numpy()
            else:
                mask = (mask_logits > 0.0)
            
            # Handle different mask shapes - squeeze out batch/channel dimensions
            if mask.ndim == 3:
                if mask.shape[0] == 1:
                    mask = mask.squeeze(0)  # Remove batch dimension
                elif mask.shape[2] == 1:
                    mask = mask.squeeze(2)  # Remove channel dimension
            elif mask.ndim == 2:
                pass  # Already correct shape
            else:
                mask = mask.squeeze()
            
            # Ensure 2D mask
            if mask.ndim != 2:
                # Take first slice if still multi-dimensional
                mask = mask[0] if mask.shape[0] == 1 else mask.squeeze()
            
            # Resize to original frame size if needed
            if mask.shape != (height, width):
                mask = cv2.resize(mask.astype(np.uint8), (width, height), 
                                interpolation=cv2.INTER_NEAREST)
            
            # Binarize to uint8 (0 or 1)
            mask = (mask > 0.5).astype(np.uint8)
            
            masks.append(mask)
        
        # Display debug overlay if enabled
        if self.debug and masks is not None and len(masks) > 0:
            overlay = self._create_overlay(frame, masks)
            cv2.imshow(self.window_name, overlay)
            cv2.waitKey(1)  # Required for window to update
        
        return masks
    
    def close(self):
        """Clean up resources."""
        # Close debug window if open
        if self.debug:
            cv2.destroyAllWindows()
        # Note: Camera cleanup should be handled by the caller
        print("SAM2Tracker closed")


if __name__ == "__main__":
    # Auto-detect object positions using colors only
    from camera import Camera
    camera = Camera(debug=False, imsize=(480, 360))
    tracker = SAM2Tracker(
        camera=camera,
        model_type="tiny",
        init_pos_object_color=[['g'], ['r']],
        object_count=2,
        debug=True,
        window_name="SAM2 Tracker"
    )
    try:
        while True:
            masks = tracker.detect_in_next_frame()  # Fetches from camera
            if tracker.debug:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        tracker.close()
        camera.close()

