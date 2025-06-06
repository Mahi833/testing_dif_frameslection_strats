#!/usr/bin/env python3
"""
Frame Extraction Module
Uses config.py for all settings
"""

import cv2
import numpy as np
import os
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import time
from config import get_config

# Get configuration
config = get_config()

class VideoWriter:
    """Video reader class"""
    def __init__(self, fname):
        self.cap = cv2.VideoCapture(fname)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
    def __len__(self):
        return self.nframes
    
    def set_to_frame(self, frame_idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    
    def read_frame(self, crop=False):
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    def close(self):
        self.cap.release()


def kmeans_sampling(cap, numframes2pick, start, stop, step=1, resizewidth=30, 
                   color=False, batchsize=100, max_iter=50):
    """DeepLabCut's K-means frame selection"""
    
    nframes = len(cap)
    startindex = int(start * nframes)
    stopindex = int(stop * nframes)
    Index = np.arange(startindex, stopindex, step, dtype=int)
    
    if len(Index) < numframes2pick:
        print(f"Not enough frames. Available: {len(Index)}, Requested: {numframes2pick}")
        return list(Index)
    
    print(f"K-means sampling from {start*100:.1f}% to {stop*100:.1f}% of video.")
    print(f"Extracting and downsampling... {len(Index)} frames from the video.")
    
    # Extract and downsample frames for clustering
    DATA = []
    print("Collecting frames for clustering...")
    
    for i in tqdm(Index):
        cap.set_to_frame(i)
        frame = cap.read_frame()
        
        if frame is not None:
            if not color:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            height, width = frame.shape[:2]
            if width > height:
                new_width = resizewidth
                new_height = int(height * resizewidth / width)
            else:
                new_height = resizewidth
                new_width = int(width * resizewidth / height)
            
            frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            DATA.append(frame_resized.flatten())
    
    if not DATA:
        print("No valid frames found!")
        return []
    
    # K-means clustering
    DATA = np.array(DATA, dtype=np.float32)
    data = DATA - DATA.mean(axis=0)
    data = data.reshape(len(DATA), -1)
    
    print(f"Running K-means clustering with {numframes2pick} clusters...")
    
    kmeans = MiniBatchKMeans(
        n_clusters=numframes2pick, 
        tol=1e-3, 
        batch_size=batchsize, 
        max_iter=max_iter,
        random_state=42
    )
    kmeans.fit(data)
    
    # Select one frame per cluster
    frames2pick = []
    for clusterid in range(numframes2pick):
        clusterids = np.where(clusterid == kmeans.labels_)[0]
        if len(clusterids) > 0:
            selected_idx = clusterids[np.random.randint(len(clusterids))]
            frames2pick.append(Index[selected_idx])
    
    return sorted(frames2pick)


def uniform_sampling(cap, numframes2pick, start, stop):
    """Uniform temporal sampling - evenly spaced frames"""
    
    nframes = len(cap)
    startindex = int(start * nframes)
    stopindex = int(stop * nframes)
    
    print(f"Uniform sampling from {start*100:.1f}% to {stop*100:.1f}% of video.")
    print(f"Selecting {numframes2pick} evenly spaced frames...")
    
    # Evenly spaced frame indices
    frame_indices = np.linspace(startindex, stopindex-1, numframes2pick, dtype=int)
    
    return sorted(frame_indices.tolist())


def extract_frames(video_path, output_folder):
    """Extract frames using method from config"""
    
    print(f"Processing video: {Path(video_path).name}")
    print(f"Method: {config['sampling_method']}")
    start_time = time.time()
    
    # Create output directory
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = VideoWriter(video_path)
    print(f"Video info: {cap.nframes} frames, {cap.fps:.1f} FPS, {cap.width}x{cap.height}")
    
    # Calculate video duration
    video_duration = cap.nframes / cap.fps if cap.fps > 0 else 0
    
    # Select sampling method
    if config['sampling_method'] == "kmeans":
        selected_frames = kmeans_sampling(
            cap, 
            config['numframes_to_extract'], 
            config['video_start_fraction'], 
            config['video_stop_fraction'], 
            **config['method_params']
        )
    elif config['sampling_method'] == "uniform":
        selected_frames = uniform_sampling(
            cap, 
            config['numframes_to_extract'], 
            config['video_start_fraction'], 
            config['video_stop_fraction']
        )
    else:
        raise ValueError(f"Unknown method: {config['sampling_method']}")
    
    if not selected_frames:
        print("No frames selected!")
        cap.close()
        return []
    
    # Save selected frames with detailed naming
    video_name = Path(video_path).stem
    saved_files = []
    
    print(f"Saving {len(selected_frames)} selected frames...")
    
    for i, frame_idx in enumerate(selected_frames):
        cap.set_to_frame(frame_idx)
        frame = cap.read_frame()
        
        if frame is not None:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Calculate timestamp for this frame
            timestamp_seconds = frame_idx / cap.fps if cap.fps > 0 else 0
            minutes = int(timestamp_seconds // 60)
            seconds = timestamp_seconds % 60
            
            # Detailed filename with timing info
            filename = (f"{video_name}_{config['sampling_method']}_"
                       f"frame_{i+1:03d}_of_{len(selected_frames):03d}_"
                       f"idx_{frame_idx:06d}_"
                       f"time_{minutes:02d}m{seconds:05.2f}s.jpg")
            
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, frame_bgr)
            saved_files.append(filepath)
    
    cap.close()
    
    end_time = time.time()
    
    # Create detailed log file
    log_path = os.path.join(output_folder, f"{video_name}_{config['sampling_method']}_extraction_log.txt")
    with open(log_path, 'w') as f:
        f.write(f"{config['sampling_method'].upper()} Frame Extraction Log\n")
        f.write(f"================================\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Video name: {video_name}\n")
        f.write(f"Method: {config['sampling_method']}\n")
        f.write(f"Video duration: {video_duration:.2f} seconds ({video_duration/60:.2f} minutes)\n")
        f.write(f"Total video frames: {cap.nframes}\n")
        f.write(f"Video FPS: {cap.fps:.2f}\n")
        f.write(f"Processing time: {end_time - start_time:.2f} seconds\n")
        f.write(f"Frames requested: {config['numframes_to_extract']}\n")
        f.write(f"Frames extracted: {len(saved_files)}\n")
        f.write(f"Video range: {config['video_start_fraction']*100:.1f}% to {config['video_stop_fraction']*100:.1f}%\n")
        f.write(f"Method parameters: {config['method_params']}\n")
        f.write(f"\nExtracted frames with timestamps:\n")
        f.write(f"{'#':<4} {'Frame Index':<12} {'Timestamp':<15} {'Filename':<50}\n")
        f.write(f"{'-'*4} {'-'*12} {'-'*15} {'-'*50}\n")
        
        for i, frame_idx in enumerate(selected_frames):
            timestamp_seconds = frame_idx / cap.fps if cap.fps > 0 else 0
            minutes = int(timestamp_seconds // 60)
            seconds = timestamp_seconds % 60
            timestamp_str = f"{minutes:02d}m{seconds:05.2f}s"
            filename = Path(saved_files[i]).name
            f.write(f"{i+1:<4} {frame_idx:<12} {timestamp_str:<15} {filename}\n")
    
    print(f"Complete! {len(saved_files)} frames saved to: {output_folder}")
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    
    return saved_files


if __name__ == "__main__":
    print(f"Starting {config['sampling_method'].upper()} Frame Extraction")
    print(f"Output directory: {config['frames_output_dir']}")
    print("="*60)
    
    # Process each video
    all_results = {}
    
    for i, video_path in enumerate(config['video_paths'], 1):
        print(f"\nProcessing Video {i}/{len(config['video_paths'])}: {Path(video_path).name}")
        
        video_name = Path(video_path).stem
        # Use timestamped output structure - each video gets its own folder within timestamp
        output_folder = os.path.join(config['frames_output_dir'], f"{video_name}_{config['sampling_method']}")
        
        try:
            saved_files = extract_frames(video_path, output_folder)
            all_results[video_name] = {
                "status": "success",
                "frames_extracted": len(saved_files),
                "output_folder": output_folder
            }
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
            all_results[video_name] = {
                "status": "failed", 
                "error": str(e)
            }
    
    # Create summary report in timestamped directory
    summary_path = os.path.join(config['frames_output_dir'], f"{config['sampling_method']}_extraction_summary.txt")
    os.makedirs(config['frames_output_dir'], exist_ok=True)
    
    with open(summary_path, 'w') as f:
        f.write(f"{config['sampling_method'].upper()} Frame Extraction Summary\n")
        f.write("="*40 + "\n\n")
        f.write(f"Method: {config['sampling_method']}\n")
        f.write(f"Parameters: {config['method_params']}\n")
        f.write(f"Timestamp: {os.path.basename(config['frames_output_dir'])}\n\n")
        
        total_frames = 0
        successful_videos = 0
        
        for video_name, result in all_results.items():
            f.write(f"Video: {video_name}\n")
            if result["status"] == "success":
                f.write(f"  Status: SUCCESS\n")
                f.write(f"  Frames extracted: {result['frames_extracted']}\n")
                f.write(f"  Output: {result['output_folder']}\n")
                total_frames += result['frames_extracted']
                successful_videos += 1
            else:
                f.write(f"  Status: FAILED\n")
                f.write(f"  Error: {result['error']}\n")
            f.write("\n")
        
        f.write(f"SUMMARY:\n")
        f.write(f"  Videos processed: {len(config['video_paths'])}\n")
        f.write(f"  Successful: {successful_videos}\n")
        f.write(f"  Failed: {len(config['video_paths']) - successful_videos}\n")
        f.write(f"  Total frames extracted: {total_frames}\n")
    
    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE!")
    print(f"Summary report: {summary_path}")
    
    # Print quick summary
    successful = sum(1 for r in all_results.values() if r["status"] == "success")
    total_frames = sum(r.get("frames_extracted", 0) for r in all_results.values())
    
    print(f"Successfully processed: {successful}/{len(config['video_paths'])} videos")
    print(f"Total frames extracted: {total_frames}")
    print(f"All outputs saved to: {config['frames_output_dir']}")