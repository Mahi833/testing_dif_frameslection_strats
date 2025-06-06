#!/usr/bin/env python3
"""
ChatGPT-4o Frame Analysis for Fall Detection
Sends extracted frames to OpenAI GPT-4o for analysis
"""

import os
import sys
import base64
import json
import glob
from pathlib import Path
from dotenv import load_dotenv
import openai
from openai import OpenAI
import time
from config import get_config

# Load environment variables
load_dotenv()

# Get configuration
config = get_config()

# ============= CONFIGURATION FROM CONFIG.PY =============

# All settings now come from config.py - no user configuration here
LATEST_FRAMES_DIR = config['latest_frames_dir']
VIDEOS_TO_ANALYZE = config['videos_to_analyze']

# Analysis Settings
FRAMES_PER_ANALYSIS = None  # Send ALL selected frames - this is the whole point!
FRAME_FORMAT = "*.jpg"    # Frame file format

# All settings from config.py
SELECTED_PROMPT = config['analysis_prompt_type']
ANALYSIS_PROMPT = config['analysis_prompts'][SELECTED_PROMPT]
TEMPERATURE = config['chatgpt_temperature']
MAX_TOKENS = config['chatgpt_max_tokens'] 
MODEL_NAME = config['chatgpt_model']
API_OUTPUT_DIR = config['api_call_output_dir']

# ============= FUNCTIONS =============

def encode_image(image_path):
    """Encode image to base64 for OpenAI API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_frames_from_folder(folder_path):
    """Get all frame files from a folder"""
    frame_files = glob.glob(os.path.join(folder_path, FRAME_FORMAT))
    return sorted(frame_files)

def analyze_frames_with_chatgpt(frame_paths, video_name):
    """Send frames to ChatGPT-4o for analysis"""
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Video: {video_name}\n\n{ANALYSIS_PROMPT}"
                }
            ]
        }
    ]
    
    # Add images to message
    for i, frame_path in enumerate(frame_paths):
        try:
            base64_image = encode_image(frame_path)
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"
                }
            })
            print(f"  Added frame {i+1}: {Path(frame_path).name}")
        except Exception as e:
            print(f"  Error encoding {frame_path}: {e}")
    
    # Make API call
    try:
        print(f"  Sending {len(frame_paths)} frames to ChatGPT-4o...")
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"  API Error: {e}")
        return f"Error: {e}"

def save_analysis_results(video_name, analysis_text, frame_paths):
    """Save analysis results to file in API output directory"""
    
    # Create API output directory
    os.makedirs(API_OUTPUT_DIR, exist_ok=True)
    
    # Save main analysis
    analysis_file = os.path.join(API_OUTPUT_DIR, f"{video_name}_analysis.txt")
    with open(analysis_file, 'w', encoding='utf-8') as f:
        f.write(f"ChatGPT Fall Detection Analysis\n")
        f.write(f"==============================\n\n")
        f.write(f"Video: {video_name}\n")
        f.write(f"Sampling method: {config['sampling_method']}\n")
        f.write(f"Prompt type: {SELECTED_PROMPT}\n")
        f.write(f"Frames analyzed: {len(frame_paths)}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Temperature: {TEMPERATURE}\n")
        f.write(f"Max tokens: {MAX_TOKENS}\n")
        f.write(f"Source frames directory: {LATEST_FRAMES_DIR}\n\n")
        
        f.write(f"Frame list:\n")
        for i, frame_path in enumerate(frame_paths):
            frame_name = Path(frame_path).name
            f.write(f"  {i+1:2d}. {frame_name}\n")
        
        f.write(f"\nAnalysis Results:\n")
        f.write(f"================\n")
        f.write(analysis_text)
        f.write(f"\n\nAnalysis completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"  Analysis saved: {analysis_file}")
    return analysis_file

def main():
    """Main execution function"""
    
    print(f"ChatGPT Frame Analysis - {config['sampling_method'].upper()} Method")
    print("=" * 50)
    
    # Check if frames directory exists
    if not LATEST_FRAMES_DIR:
        print(f"ERROR: No frames directory found for {config['sampling_method']} method!")
        print("Please run frame_extractor.py first to generate frames.")
        return
    
    if not os.path.exists(LATEST_FRAMES_DIR):
        print(f"ERROR: Frames directory not found: {LATEST_FRAMES_DIR}")
        print("Please run frame_extractor.py first to generate frames.")
        return
    
    print(f"Using frames from: {LATEST_FRAMES_DIR}")
    print(f"Selected prompt: {SELECTED_PROMPT}")
    print(f"Frames per analysis: ALL {config['sampling_method']} selected frames")
    print()
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY not found in environment variables!")
        print("Make sure your .env file contains the API key.")
        return
    
    all_results = {}
    
    # Process each video folder
    for video_folder in VIDEOS_TO_ANALYZE:
        print(f"Processing: {video_folder}")
        
        # Use latest frames directory structure
        video_name = video_folder.replace("_frames", "")  # edited_mmc1_frames -> edited_mmc1
        folder_path = os.path.join(LATEST_FRAMES_DIR, f"{video_name}_{config['sampling_method']}")
        
        if not os.path.exists(folder_path):
            print(f"  Folder not found: {folder_path}")
            all_results[video_folder] = {"status": "folder_not_found"}
            continue
        
        # Get frame files
        frame_files = get_frames_from_folder(folder_path)
        
        if not frame_files:
            print(f"  No frames found in {folder_path}")
            all_results[video_folder] = {"status": "no_frames"}
            continue
        
        print(f"  Found {len(frame_files)} {config['sampling_method']} selected frames")
        
        # Use ALL selected frames - that's the whole point of this method!
        frames_to_analyze = frame_files
        
        try:
            # Analyze with ChatGPT
            analysis = analyze_frames_with_chatgpt(frames_to_analyze, video_folder)
            
            # Save results to API output directory
            analysis_file = save_analysis_results(video_folder, analysis, frames_to_analyze)
            
            # Also print results to terminal
            print("=" * 60)
            print(f"ANALYSIS RESULTS FOR: {video_folder}")
            print("=" * 60)
            print(analysis)
            print("=" * 60)
            print()
            
            all_results[video_folder] = {
                "status": "success",
                "frames_analyzed": len(frames_to_analyze),
                "analysis_file": analysis_file
            }
            
            print(f"  Analysis complete!")
            
        except Exception as e:
            print(f"  Error: {e}")
            all_results[video_folder] = {"status": "error", "error": str(e)}
        
        print()
    
    # Create summary report in timestamped API output directory
    summary_path = os.path.join(API_OUTPUT_DIR, f"{config['sampling_method']}_analysis_summary.txt")
    os.makedirs(API_OUTPUT_DIR, exist_ok=True)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("ChatGPT Fall Detection Analysis Summary\n")
        f.write("======================================\n\n")
        f.write(f"Sampling method: {config['sampling_method']}\n")
        f.write(f"Prompt type: {SELECTED_PROMPT}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Temperature: {TEMPERATURE}\n")
        f.write(f"Max tokens: {MAX_TOKENS}\n")
        f.write(f"Analysis date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source frames: {LATEST_FRAMES_DIR}\n")
        f.write(f"Timestamp: {os.path.basename(API_OUTPUT_DIR)}\n\n")
        
        successful = 0
        total_frames = 0
        
        for video, result in all_results.items():
            f.write(f"Video: {video}\n")
            if result["status"] == "success":
                f.write(f"  Status: SUCCESS\n")
                f.write(f"  Frames analyzed: {result['frames_analyzed']}\n")
                f.write(f"  Analysis file: {result['analysis_file']}\n")
                successful += 1
                total_frames += result['frames_analyzed']
            else:
                f.write(f"  Status: FAILED\n")
                if 'error' in result:
                    f.write(f"  Error: {result['error']}\n")
            f.write("\n")
        
        f.write(f"SUMMARY:\n")
        f.write(f"  Videos processed: {len(VIDEOS_TO_ANALYZE)}\n")
        f.write(f"  Successful: {successful}\n")
        f.write(f"  Failed: {len(VIDEOS_TO_ANALYZE) - successful}\n")
        f.write(f"  Total frames analyzed: {total_frames}\n")
    
    # Print final summary to terminal
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Method: {config['sampling_method']}")
    print(f"Configuration: {SELECTED_PROMPT}")
    print(f"Model: {MODEL_NAME}")
    print(f"Frames per video: ALL {config['sampling_method']} selected frames")
    print()
    
    successful = 0
    total_frames = 0
    
    for video, result in all_results.items():
        print(f"Video: {video}")
        if result["status"] == "success":
            print(f"  Status: SUCCESS")
            print(f"  Frames analyzed: {result['frames_analyzed']}")
            successful += 1
            total_frames += result['frames_analyzed']
        else:
            print(f"  Status: {result['status'].upper()}")
            if 'error' in result:
                print(f"  Error: {result['error']}")
        print()
    
    print("=" * 60)
    print("ANALYSIS COMPLETE!")
    print(f"Successfully analyzed: {successful}/{len(VIDEOS_TO_ANALYZE)} videos")
    print(f"Total frames analyzed: {total_frames}")
    print(f"Summary report: {summary_path}")
    print(f"All analysis files saved to: {API_OUTPUT_DIR}")

if __name__ == "__main__":
    main()