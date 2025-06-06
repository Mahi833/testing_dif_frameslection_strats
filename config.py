#!/usr/bin/env python3
"""
User Configuration Module
Set all your preferences here once, used by all other modules
"""

# ============= USER CONFIGURABLE SETTINGS =============

# Frame Sampling Method Selection
SAMPLING_METHOD = "uniform"  # Options: "kmeans", "uniform"

# Video File Paths
VIDEO_PATHS = [
    r"C:\Users\Mahan\Desktop\steves_fall_vids\edited_vids\edited_mmc1.mp4",
    r"C:\Users\Mahan\Desktop\steves_fall_vids\edited_vids\edited_mmc2.mp4", 
    r"C:\Users\Mahan\Desktop\steves_fall_vids\edited_vids\edited_mmc3.mp4",
    r"C:\Users\Mahan\Desktop\steves_fall_vids\edited_vids\edited_mmc4.mp4",
    r"C:\Users\Mahan\Desktop\steves_fall_vids\edited_vids\edited_mmc5.mp4"
]

# Output Directories - FIXED PATHS
BASE_OUTPUT_DIR = r"C:\Users\Mahan\Desktop\Test frame selection options (1-5 videos)\output"
FRAMES_OUTPUT_DIR = f"{BASE_OUTPUT_DIR}\\frames"
API_CALL_OUTPUT_DIR = f"{BASE_OUTPUT_DIR}\\api_call_output"

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

# ============= DERIVED SETTINGS (DON'T CHANGE) =============

# Automatically set paths based on method
CHATGPT_ANALYSIS_DIR = API_CALL_OUTPUT_DIR

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

def get_config():
    """Return all configuration as dictionary"""
    return {
        "sampling_method": SAMPLING_METHOD,
        "video_paths": VIDEO_PATHS,
        "frames_output_dir": FRAMES_OUTPUT_DIR,
        "api_call_output_dir": API_CALL_OUTPUT_DIR,
        "chatgpt_analysis_dir": CHATGPT_ANALYSIS_DIR,
        "videos_to_analyze": VIDEOS_TO_ANALYZE,
        "numframes_to_extract": NUMFRAMES_TO_EXTRACT,
        "video_start_fraction": VIDEO_START_FRACTION,
        "video_stop_fraction": VIDEO_STOP_FRACTION,
        "method_params": METHOD_PARAMS,
        "openai_api_key_file": OPENAI_API_KEY_FILE,
        "chatgpt_model": CHATGPT_MODEL,
        "chatgpt_temperature": CHATGPT_TEMPERATURE,
        "chatgpt_max_tokens": CHATGPT_MAX_TOKENS,
        "analysis_prompt_type": ANALYSIS_PROMPT_TYPE
    }

def print_config():
    """Print current configuration"""
    print("Current Configuration:")
    print("=" * 40)
    print(f"Sampling Method: {SAMPLING_METHOD}")
    print(f"Videos to process: {len(VIDEO_PATHS)}")
    print(f"Frames per video: {NUMFRAMES_TO_EXTRACT}")
    print(f"Frames output: {FRAMES_OUTPUT_DIR}")
    print(f"API output: {API_CALL_OUTPUT_DIR}")
    print(f"Method parameters: {METHOD_PARAMS}")
    print(f"Analysis prompt: {ANALYSIS_PROMPT_TYPE}")
    print("=" * 40)

if __name__ == "__main__":
    print_config()