import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
import os
import glob
import json

def get_video_frames(video_path):
    """Read all frames from a video file into a list of RGB arrays."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def extract_color_centroid(frames, target_color, tolerance=10):
    """
    For each frame, find contours matching `target_color` ± tolerance,
    return (cx, cy) arrays of the largest-contour centroid.
    """
    centroids_x, centroids_y = [], []
    lower_bound = np.array([max(0, int(c) - tolerance) for c in target_color], dtype=np.uint8)
    upper_bound = np.array([min(255, int(c) + tolerance) for c in target_color], dtype=np.uint8)
    
    for frame in frames:
        mask = cv2.inRange(frame, lower_bound, upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            else:
                cX, cY = np.nan, np.nan
        else:
            cX, cY = np.nan, np.nan
        centroids_x.append(cX)
        centroids_y.append(cY)
    return np.array(centroids_x), np.array(centroids_y)

def extract_mask_area(frames, target_color, tolerance=40):
    """
    For each frame, compute the pixel area of the region matching target_color.
    Returns a 1D array of area values over time.
    """
    areas = []
    lower_bound = np.array([max(0, int(c) - tolerance) for c in target_color], dtype=np.uint8)
    upper_bound = np.array([min(255, int(c) + tolerance) for c in target_color], dtype=np.uint8)
    for frame in frames:
        mask = cv2.inRange(frame, lower_bound, upper_bound)
        areas.append(np.sum(mask > 0))
    return np.array(areas, dtype=float)

def extract_optical_flow_magnitude(frames):
    """
    Compute dense optical flow (Farneback) between consecutive frames.
    Returns a 1D array of mean flow magnitude per frame (first frame = 0).
    """
    magnitudes = [0.0]
    gray_prev = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    for frame in frames[1:]:
        gray_curr = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray_prev, gray_curr, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        magnitudes.append(float(np.mean(mag)))
        gray_prev = gray_curr
    return np.array(magnitudes)

def lowpass_filter(data, cutoff, fs, order=5):
    """Apply a Butterworth low-pass filter to a 1D signal."""
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low', analog=False)
    return filtfilt(b, a, data)

def save_clip(frames, start_frame, end_frame, output_path, fps=30):
    """Write a sub-clip of frames[start_frame:end_frame] to an MP4 file."""
    if start_frame >= end_frame or start_frame < 0 or end_frame > len(frames):
        return
    clip_frames = frames[start_frame:end_frame]
    height, width, _ = clip_frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in clip_frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

def build_composite_signal(frames, target_color):
    """
    Option B: Build a composite motion signal from multiple sources:
      - cy (Y centroid): captures vertical motion (squats, leg raises)
      - cx (X centroid): captures horizontal motion (arm swings, pendulum)
      - optical_flow:    captures any motion regardless of direction
      - mask_area:       captures body expansion/contraction (squats, breathing)
    Selects the signal with highest variance as the primary segmentation signal.
    All signals are normalised to [0, 1] before comparison.
    """
    cx, cy = extract_color_centroid(frames, target_color, tolerance=40)
    cy_s = pd.Series(cy).interpolate().bfill().ffill().values
    cx_s = pd.Series(cx).interpolate().bfill().ffill().values
    area   = extract_mask_area(frames, target_color, tolerance=40)
    flow   = extract_optical_flow_magnitude(frames)

    def _norm(sig):
        r = sig.max() - sig.min()
        return (sig - sig.min()) / r if r > 0 else sig - sig.mean()

    signals = {
        "cy":   cy_s,
        "cx":   cx_s,
        "area": area,
        "flow": flow,
    }

    best_name, best_sig, max_var = None, None, -1
    for name, sig in signals.items():
        v = np.var(_norm(sig))
        if v > max_var:
            max_var = v
            best_name = name
            best_sig = sig

    return best_sig, best_name

def process_video(video_path):
    """
    Full pipeline for one video:
      1. Load frames
      2. Find dominant semantic color (non-black pixel with highest frequency)
      3. Build composite motion signal (Option B: cy, cx, area, optical flow)
      4. Smooth + valley detection → rep boundaries
      5. Save per-rep MP4 clip + .npy feature file + temporal_segments.json
    """
    print(f"Processing {video_path}...")
    frames = get_video_frames(video_path)
    if not frames:
        print(f"  -> No frames loaded, skipping.")
        return
        
    parts = video_path.split(os.sep)
    subject_id, exercise_id = parts[-3], parts[-2]
    out_dir = os.path.join("reps", subject_id, exercise_id)
    os.makedirs(out_dir, exist_ok=True)
    
    # --- 1. Find dominant foreground color from first frame ---
    first_frame = frames[0]
    pixels = first_frame.reshape(-1, 3)
    non_black_pixels = pixels[np.sum(pixels, axis=1) > 30]
    
    if len(non_black_pixels) == 0:
        print(f"  -> Frame is all black, skipping.")
        return
        
    unique_colors, counts = np.unique(non_black_pixels, axis=0, return_counts=True)
    sorted_indices = np.argsort(-counts)
    unique_colors = unique_colors[sorted_indices]

    # --- 2. Try top-5 colors; for each, build composite signal; pick best ---
    best_color, best_signal, best_signal_name, max_variance = None, None, None, -1
    
    for i in range(min(5, len(unique_colors))):
        color = unique_colors[i]
        sig, sig_name = build_composite_signal(frames, color)
        variance = np.var(sig)
        if variance > max_variance:
            max_variance = variance
            best_color = color
            best_signal = sig
            best_signal_name = sig_name
            
    if best_signal is None:
        print(f"  -> Could not extract a valid signal, skipping.")
        return

    print(f"  -> Best signal: '{best_signal_name}' (variance={max_variance:.2f})")
        
    # --- 3. Temporal Segmentation ---
    smoothed = lowpass_filter(best_signal, cutoff=1.5, fs=30, order=4)

    # Invert signal for valley-as-boundary detection where appropriate
    # (valleys in Y = bottom of motion arc = rep boundary)
    # We detect valleys on the smoothed signal
    valleys, props = find_peaks(-smoothed, prominence=5, distance=10)

    # Fallback: if very few valleys detected, relax parameters further
    if len(valleys) < 2:
        valleys, props = find_peaks(-smoothed, prominence=2, distance=8)

    # --- 4. Save clips and features ---
    if len(valleys) >= 2:
        rep_segments = []
        for i in range(len(valleys) - 1):
            start_f, end_f = int(valleys[i]), int(valleys[i+1])
            if end_f - start_f > 20:  # relaxed from 30 to 20 frames (~0.67s)
                output_path = os.path.join(out_dir, f"rep_{i+1}.mp4")
                feature_path = os.path.join(out_dir, f"rep_{i+1}_features.npy")
                
                save_clip(frames, start_f, end_f, output_path)
                np.save(feature_path, smoothed[start_f:end_f])
                
                rep_segments.append({
                    "rep_idx": i+1, 
                    "start_frame": start_f, 
                    "end_frame": end_f,
                    "signal_type": best_signal_name,
                    "feature_file": f"rep_{i+1}_features.npy"
                })
                
        with open(os.path.join(out_dir, "temporal_segments.json"), "w") as f:
            json.dump(rep_segments, f, indent=4)
        print(f"  -> Saved {len(rep_segments)} reps to {out_dir}")
    else:
        print(f"  -> Only {len(valleys)} valleys detected (need ≥2). No reps saved.")
        print(f"     Hint: signal variance={max_variance:.2f}, signal_type={best_signal_name}")
        # Save empty segments file so scoring_model skips gracefully
        with open(os.path.join(out_dir, "temporal_segments.json"), "w") as f:
            json.dump([], f)

if __name__ == "__main__":
    videos = sorted(glob.glob("semantic_segmentation/*/*/cam0.mp4"))
    print(f"Found {len(videos)} videos to process.")
    for idx, video in enumerate(videos):
        print(f"[{idx+1}/{len(videos)}]", end=" ")
        process_video(video)
