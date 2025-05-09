# ASL Detector

**ASL_Detector** is a LSTM model that detects American Sign Language (ASL) gestures and translates them into English using Gemini or Nebius API.

## Environment Setup

###  Windows (without Conda)

bash
```
py -3.12 -m venv venv312

(
  Or create .venv outside of folder with the help of VSC IDE:
  - Navigate to .ipynb file
  - On the top right corner, there's a kernel box that displays either your current Python version or the text: "Select Kernel"
  - Click on the kernel box, click on "Select Another Kernel"
  - Click on "Python Environments..."
  - Click on "Create Python Environment"
  - Click on Venv (.venv), choose the Python version that's compatible (3.9 - 3.12)
)

.\venv312\Scripts\activate

(or ..\.venv\Scripts\activate)

pip install -r requirements.txt
```

### MacOS 

```
python3.11 -m venv venv311

source venv311/bin/activate

pip install -r requirements.txt
```

Delete cv2.CAP_DSHOW to prevent problems

## How to use it
The main notebook in this project is:
asl_alphabet_new.ipynb

It contains 4 main sections:
  1. Collecting Data – Capture ASL hand signs using your webcam.
  2. Create Dataset – Preprocess and structure the collected data.
  3. Train Model – Train a Convolutional Neural Network (CNN) using PyTorch.
  4. Live Test – Use your webcam to test the trained model in real-time.

If you want to use the translation part, remember to .env file with your API KEY. It should look like this `GEMINI_API_KEY=<your_key>`
Run `new_live_test_3D.py` to try the translation from ASL to English.


# Running `live_save_test.py`

0. Activate virtual environment

1. Recording video with realtime ASL recognition, then annotate (with both raw and result videos saved afterwards): 

`py .\live_save_test.py --realtime`

2. Annotating existing video:
`python live_save_test.py --input raw_videos/test2.mp4 --output_dir ./annotated_videos`

3. Recording video without realtime ASL recognition, then annotate (with both raw and result videos saved afterwards):
`py .\live_save_test.py --record`

Raw / Input videos will / must be stored in `raw_videos`
Result videos will be generated in `annotated_videos`