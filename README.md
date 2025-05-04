# ASL Detector

**ASL_Detector** is a deep learning-based application that detects American Sign Language (ASL) gestures and translates them into English using Gemini or Nebius API.

## ⚙Environment Setup

###  Windows (without Conda)
bash
py -3.12 -m venv venv312
.\venv312\Scripts\activate
pip install google-generativeai
pip install python-dotenv
pip install opencv-python
pip install mediapipe
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install openai

### MacOS 

python3.11 -m venv venv311
source venv311/bin/activate
pip install -r requirements.txt

## How to use it
The main notebook in this project is:
asl_alphabet_new.ipynb

It contains 4 main sections:
  1. Collecting Data – Capture ASL hand signs using your webcam.
  2. Create Dataset – Preprocess and structure the collected data.
  3. Train Model – Train a Convolutional Neural Network (CNN) using PyTorch.
  4. Live Test – Use your webcam to test the trained model in real-time.



