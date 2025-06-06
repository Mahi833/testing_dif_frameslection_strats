#!/usr/bin/env python3
"""
User Configuration Module
Set all your preferences here once, used by all other modules
"""

import os
from datetime import datetime

# ============= USER CONFIGURABLE SETTINGS =============

# Frame Sampling Method Selection
SAMPLING_METHOD = "kmeans"  # Options: "kmeans", "uniform"

# Video File Paths
VIDEO_PATHS = [
    r"C:\Users\Mahan\Desktop\steves_fall_vids\edited_vids\edited_mmc1.mp4",
    r"C:\Users\Mahan\Desktop\steves_fall_vids\edited_vids\edited_mmc2.mp4", 
    r"C:\Users\Mahan\Desktop\steves_fall_vids\edited_vids\edited_mmc3.mp4",
    r"C:\Users\Mahan\Desktop\steves_fall_vids\edited_vids\edited_mmc4.mp4",
    r"C:\Users\Mahan\Desktop\steves_fall_vids\edited_vids\edited_mmc5.mp4"
]

# Output Directories - UPDATED FOR TIMESTAMPED FOLDERS
BASE_OUTPUT_DIR = r"C:\Users\Mahan\Desktop\Test frame selection options (1-5 videos)\output"

# Frame Extraction Settings
NUMFRAMES_TO_EXTRACT = 20
VIDEO_START_FRACTION = 0.0  # 0.0 = beginning of video
VIDEO_STOP_FRACTION = 1.0   # 1.0 = end of video

# K-means Specific Parameters
KMEANS_PARAMS = {
    "step": 1,
    "resizewidth": 30,
    "color": False,
    "batchsize": 100,
    "max_iter": 50
}

# OpenAI API Settings
OPENAI_API_KEY_FILE = ".env"  # Path to your .env file
CHATGPT_MODEL = "gpt-4o"
CHATGPT_TEMPERATURE = 0.1
CHATGPT_MAX_TOKENS = 2000

# Analysis Prompt Selection
ANALYSIS_PROMPT_TYPE = "comprehensive_fall_analysis"  # Options: "comprehensive_fall_analysis", "simple_fall_detection"

# Analysis Prompts - centralized here for single source of truth
ANALYSIS_PROMPTS = {
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

# ============= DERIVED SETTINGS (DON'T CHANGE) =============

# Video folder names for analysis
VIDEOS_TO_ANALYZE = [
    "edited_mmc1_frames",
    "edited_mmc2_frames", 
    "edited_mmc3_frames",
    "edited_mmc4_frames",
    "edited_mmc5_frames"
]

# Method-specific parameters
if SAMPLING_METHOD == "kmeans":
    METHOD_PARAMS = KMEANS_PARAMS
else:  # uniform
    METHOD_PARAMS = {}

def get_current_timestamp():
    """Generate compact timestamp for folder naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_timestamped_output_dirs():
    """Generate timestamped output directories for current run"""
    timestamp = get_current_timestamp()
    
    # Method name for folder (kmeans -> kmean for consistency with your request)
    method_name = "kmean" if SAMPLING_METHOD == "kmeans" else SAMPLING_METHOD
    
    frames_dir = os.path.join(BASE_OUTPUT_DIR, "frames", f"{method_name}_frames", timestamp)
    api_dir = os.path.join(BASE_OUTPUT_DIR, "api_call_output", method_name, timestamp)
    
    return frames_dir, api_dir

def find_latest_frames_dir():
    """Find the most recent frames directory for the current sampling method"""
    method_name = "kmean" if SAMPLING_METHOD == "kmeans" else SAMPLING_METHOD
    frames_base = os.path.join(BASE_OUTPUT_DIR, "frames", f"{method_name}_frames")
    
    if not os.path.exists(frames_base):
        return None
    
    # Get all timestamp directories and find the latest one
    timestamp_dirs = [d for d in os.listdir(frames_base) 
                     if os.path.isdir(os.path.join(frames_base, d))]
    
    if not timestamp_dirs:
        return None
    
    # Sort by timestamp (latest first)
    timestamp_dirs.sort(reverse=True)
    latest_timestamp = timestamp_dirs[0]
    
    return os.path.join(frames_base, latest_timestamp)

def get_config():
    """Return all configuration as dictionary"""
    frames_dir, api_dir = get_timestamped_output_dirs()
    
    return {
        "sampling_method": SAMPLING_METHOD,
        "video_paths": VIDEO_PATHS,
        "frames_output_dir": frames_dir,
        "api_call_output_dir": api_dir,
        "videos_to_analyze": VIDEOS_TO_ANALYZE,
        "numframes_to_extract": NUMFRAMES_TO_EXTRACT,
        "video_start_fraction": VIDEO_START_FRACTION,
        "video_stop_fraction": VIDEO_STOP_FRACTION,
        "method_params": METHOD_PARAMS,
        "openai_api_key_file": OPENAI_API_KEY_FILE,
        "chatgpt_model": CHATGPT_MODEL,
        "chatgpt_temperature": CHATGPT_TEMPERATURE,
        "chatgpt_max_tokens": CHATGPT_MAX_TOKENS,
        "analysis_prompt_type": ANALYSIS_PROMPT_TYPE,
        "analysis_prompts": ANALYSIS_PROMPTS,
        "latest_frames_dir": find_latest_frames_dir()
    }

def print_config():
    """Print current configuration"""
    config = get_config()
    print("Current Configuration:")
    print("=" * 40)
    print(f"Sampling Method: {SAMPLING_METHOD}")
    print(f"Videos to process: {len(VIDEO_PATHS)}")
    print(f"Frames per video: {NUMFRAMES_TO_EXTRACT}")
    print(f"Frames output: {config['frames_output_dir']}")
    print(f"API output: {config['api_call_output_dir']}")
    print(f"Method parameters: {METHOD_PARAMS}")
    print(f"Analysis prompt: {ANALYSIS_PROMPT_TYPE}")
    if config['latest_frames_dir']:
        print(f"Latest frames found: {config['latest_frames_dir']}")
    else:
        print("No previous frames found for this method")
    print("=" * 40)

if __name__ == "__main__":
    print_config()