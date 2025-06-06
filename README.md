# **Testing Different Frame Selection Strategies**

**Comparing uniform vs K-means frame sampling for fall detection using GPT-4o.**

---

## **What This Does**

Takes fall videos, extracts frames using different methods, sends them to GPT-4o for analysis, and compares which sampling strategy works better for detecting falls.

### **Two Methods Tested**
- **Uniform**: evenly spaced frames  
- **K-means**: clustering to pick diverse frames

---

## **Quick Setup**

1. **Put your OpenAI key in `.env` file:**
OPENAI_API_KEY=your_key_here


2. **Update video paths in `config.py`**

3. **Set SAMPLING_METHOD = "uniform" or "kmeans" in config.py**
4. **run 'frame_extrator.py'**
5. **finally, run 'chatgpt_analyzer.py'**

## **File Structure**

```text
testing_dif_frameslection_strats/
├── config.py              # All settings here
├── frame_extractor.py     # Extracts frames
├── chatgpt_analyzer.py    # Sends to GPT-4o
├── .env                   # Your OpenAI API key
├── .gitignore             # Ignores output data
└── output/                # Results (auto-organized by timestamp)
    ├── frames/
    │   ├── uniform_frames/
    │   │   └── 20250605_212124/
    │   │       ├── edited_mmc1_uniform/
    │   │       ├── edited_mmc2_uniform/
    │   │       └── ...
    │   └── kmean_frames/
    │       └── 20250605_214500/
    │           ├── edited_mmc1_kmeans/
    │           ├── edited_mmc2_kmeans/
    │           └── ...
    └── api_call_output/
        ├── uniform/
        │   └── 20250605_212258/
        │       ├── edited_mmc1_frames_analysis.txt
        │       ├── edited_mmc2_frames_analysis.txt
        │       └── uniform_analysis_summary.txt
        └── kmean/
            └── 20250605_214730/
                ├── edited_mmc1_frames_analysis.txt
                ├── edited_mmc2_frames_analysis.txt
                └── kmean_analysis_summary.txt
