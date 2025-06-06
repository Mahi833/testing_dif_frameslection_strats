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

# ============= USER CONFIGURABLE SETTINGS =============

# Frame Sampling Method Selection
SAMPLING_METHOD = "kmeans"  # Options: "uniform", "random", "kmeans", "entropy", "optical_flow", "edge_density"

# Method-specific parameters
SAMPLING_PARAMS = {
    "uniform": {},
    "random": {"seed": 42},
    "kmeans": {"step": 1, "resizewidth": 30, "color": False, "batchsize": 100, "max_iter": 50},
    "entropy": {"step": 1, "resizewidth": 64},
    "optical_flow": {"step": 1},
    "edge_density": {"step": 1}
}

# Frame Input Settings
FRAMES_BASE_DIR = config['frames_output_dir']
OUTPUT_FOLDER_NAME = f"{SAMPLING_METHOD}_strategy"  # Will create folder based on method
VIDEOS_TO_ANALYZE = [
    "edited_mmc1_frames",
    "edited_mmc2_frames", 
    "edited_mmc3_frames",
    "edited_mmc4_frames",
    "edited_mmc5_frames"
]

# Analysis Settings
FRAMES_PER_ANALYSIS = None  # Send ALL K-means selected frames - this is the whole point!
FRAME_FORMAT = "*.jpg"    # Frame file format

# Prompt Options for Fall Detection
PROMPTS = {
    "comprehensive_fall_analysis": """FOCUS: Look for ELDERLY PERSON in frames. Ignore background, furniture, other people unless directly involved in fall.

FRAME ANALYSIS PRIORITY:
1. Locate the PRIMARY SUBJECT (elderly person)
2. Track their body position across frames
3. Identify any loss of balance or ground contact
4. Determine cause and impact points

VISUAL CUES TO IDENTIFY FALLS:
- Person transitioning from upright to lower position
- Body tilting beyond normal balance range
- Arms reaching out for support/protection
- Knees buckling or legs giving way
- Person on ground/floor when previously standing
- Protective responses (hands out, bracing)
- Balance recovery attempts

ANALYZE THESE FRAMES FOR THE ELDERLY PERSON ONLY:

FALL STATUS: [NONE/PRE-FALL/ACTIVE-FALL/POST-FALL] (X%)
- NONE: Person stable, normal activities
- PRE-FALL: Loss of balance, reaching for support, unsteady
- ACTIVE-FALL: Person mid-fall, losing contact with ground
- POST-FALL: Person on ground, attempting to get up

FALL CAUSE: [Select ONE with confidence %]
- SLIP: Foot sliding/loss of traction (X%)
- TRIP: Foot collision with object/person/own foot (X%)  
- HIT: Body above knee collides with object/person (X%)
- SYNCOPE: Sudden loss of consciousness/fainting (X%)
- SEIZURE: Neurological episode causing fall (X%)
- ORTHOSTATIC: Blood pressure drop causing dizziness (X%)
- MUSCLE_WEAKNESS: Leg/core strength failure (X%)
- BALANCE_LOSS: Vestibular/proprioceptive failure (X%)
- MEDICATION: Drug-induced impairment (X%)
- CARDIAC: Heart rhythm/output problem (X%)
- ENVIRONMENTAL: Poor lighting/surfaces/obstacles (X%)
- BEHAVIORAL: Risk-taking/unsafe activity (X%)
- UNKNOWN: Cause not determinable (X%)

BODY CONTACT: [First impact point with confidence %]
- HEAD: Head/face first contact (X%)
- SHOULDER: Shoulder first contact (X%)
- ARM: Arm/hand/elbow first contact (X%)
- TORSO: Chest/abdomen first contact (X%)
- HIP: Hip/pelvis first contact (X%)
- BACK: Back/spine first contact (X%)
- KNEE: Knee first contact (X%)
- MULTIPLE: Multiple simultaneous contact (X%)
- NONE: No ground contact visible (X%)

INTRINSIC/EXTRINSIC: [INTRINSIC/EXTRINSIC/MIXED] (X%)
- INTRINSIC: Person's internal factors (health, medication, weakness)
- EXTRINSIC: Environmental factors (slippery floor, obstacles, lighting)
- MIXED: Combination of both factors

INJURY RISK: [LOW/MEDIUM/HIGH] (X%)
- LOW: Soft landing, slow fall, protective response successful
- MEDIUM: Moderate impact, some protection, controlled descent
- HIGH: Hard impact, no protection, head/hip contact, fast fall

REASON: [One sentence explanation max 15 words]

IGNORE: Other people, background objects, furniture unless causing the fall.
FOCUS ONLY: The elderly person and their movement/position changes.""",
    
    "simple_fall_detection": """Look at these frames. Answer in this exact format:

FALL: YES/NO (X%)
CAUSE: [SLIP/TRIP/HIT/SYNCOPE/WEAKNESS/BALANCE/UNKNOWN] (X%)
CONTACT: [HEAD/SHOULDER/ARM/HIP/BACK/KNEE/NONE] (X%)
TYPE: [INTRINSIC/EXTRINSIC] (X%)

One word answers. Percentages required."""
}

# Select which prompt to use from config
SELECTED_PROMPT = config['analysis_prompt_type']
ANALYSIS_PROMPT = PROMPTS[SELECTED_PROMPT]

# AI Model Settings
TEMPERATURE = config['chatgpt_temperature']
MAX_TOKENS = config['chatgpt_max_tokens'] 
MODEL_NAME = config['chatgpt_model']

# Output Settings - no file saving
SAVE_DETAILED_LOGS = False
IMAGE_QUALITY = 95

# API Limits
MAX_INPUT_TOKENS = 128000
MAX_OUTPUT_TOKENS = 4096

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
        f.write(f"Max tokens: {MAX_TOKENS}\n\n")
        
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
    print(f"Input directory: {FRAMES_BASE_DIR}")
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
        
        # FIXED: Use correct path structure that matches frame_extractor output
        video_name = video_folder.replace("_frames", "")  # edited_mmc1_frames -> edited_mmc1
        folder_path = os.path.join(FRAMES_BASE_DIR, f"{video_name}_{config['sampling_method']}")
        
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
        
        print(f"  Found {len(frame_files)} K-means selected frames")
        
        # Use ALL K-means selected frames - that's the whole point of this method!
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
    
    # Create summary report in API output directory
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
        f.write(f"Analysis date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
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