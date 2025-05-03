# ASL_Detector

### WITHOUT CONDA:
py -3.12 -m venv venv312
.\venv312\Scripts\activate
pip install google-generativeai
pip install python-dotenv
pip install opencv-python
pip install mediapipe
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install openai

#### MAC USER
python3.12 -m venv venv312
source venv312/bin/activate
pip install -r requirements.txt

### WITH CONDA:
Environment setup: 
conda create -n sign_language python=3.11
pip install -r requirements.txt
