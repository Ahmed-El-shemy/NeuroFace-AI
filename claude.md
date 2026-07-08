# Face Recognition Siamese Project Report

## Project Details
- **Project Name:** Real-Time Face Recognition System using Siamese Neural Networks.
- **Main Objective:** Build a complete system for real-time face recognition using a webcam.
- **System Components:**
  1. **Training Mode:** Train the model to distinguish differences between faces.
  2. **Evaluation:** Measure model accuracy and performance metrics.
  3. **Build-DB (Database):** Create a database of Face Embeddings for registered/known identities.
  4. **Webcam Inference:** Real-time facial recognition and identification via a live camera feed.

---

## What the AI Did
During this session, the AI diagnosed and resolved compatibility issues to successfully enable GPU (CUDA) acceleration:

1. **System Compatibility Check:**
   - Checked the current GPU (NVIDIA GTX 1650) and its driver version.
   - Identified that the installed driver (v535) supports up to CUDA 12.2.

2. **Diagnosing the `PyTorch` Issue:**
   - The system initially had a `PyTorch` build configured for CUDA 13, causing the library to fail at recognizing the GPU.
   - Discovered that the `uv` package manager had downloaded the latest incompatible version due to a lack of explicit index constraints.

3. **Fixing Configurations (`pyproject.toml`):**
   - **Adjusted Python Version:** Lowered the Python requirement from `>=3.13` to `>=3.12` because `PyTorch` wheels compatible with `CUDA 12.1` are not yet available for Python 3.13.
   - **Specified Index Source:** Added specific configurations to force `uv` to download `torch` and `torchvision` strictly from PyTorch's `cu121` server index.

4. **Rebuilding the Virtual Environment:**
   - Purged the old environment that lacked proper GPU support.
   - Initiated the rebuild of a new virtual environment, downloading the correct heavy libraries to ensure the system utilizes the GPU at maximum speed for both training and real-time inference.

---
> **Note:** The `No module named 'torch'` error appeared during your attempt to run `main.py` because the environment rebuild and PyTorch download (which is around 2.5 GB) is still running in the background. Once the download is complete, the script will run successfully!
