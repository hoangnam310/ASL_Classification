# ASL_Detector

<<<<<<< HEAD
### WITHOUT CONDA:
=======
### WITHOUT CONDA ON WINDOWS:
>>>>>>> 8801dc7a1a3bc404ef63c336ba814a4b0f8dc521
py -3.12 -m venv venv312
.\venv312\Scripts\activate
pip install google-generativeai
pip install python-dotenv
pip install opencv-python
pip install mediapipe
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install openai

<<<<<<< HEAD
#### MAC USER
python3.12 -m venv venv312
source venv312/bin/activate
pip install -r requirements.txt

=======
>>>>>>> 8801dc7a1a3bc404ef63c336ba814a4b0f8dc521
### WITH CONDA:
Environment setup: 
conda create -n sign_language python=3.11
pip install -r requirements.txt
