import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the API key
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Create a model instance
# model = genai.GenerativeModel("gemini-pro")
model = genai.GenerativeModel("gemini-1.5-flash")  # Or "gemini-1.0-pro", or "gemini-1.5-flash", or "gemini-1.5-pro"


def analyze_asl_gemini(text):
    """
    Analyze ASL space-separated letters and convert to meaningful English sentences using the Gemini API.

    Args:
        text: Space-separated ASL letters/characters

    Returns:
        Analyzed text as complete sentences
    """
    prompt = (
        "You are an expert at accurately reconstructing English sentences from ASL fingerspelled letters. "
        "Your job is to return only the most likely sentence(s) the signer intended, based on the detected letter sequence. "
        "Do NOT be creative or infer unrelated phrases. "
        "You may only expand common abbreviations and acronyms into full words. "
        "Here are some common examples:\n"
        "- ILU -> I love you\n"
        "- ILY -> I love you\n"
        "- SW -> Software\n"
        "- CS -> Computer Science\n"
        "- AI -> Artificial Intelligence\n"
        "- USF -> University of South Florida\n"
        "- NYC -> New York City\n"
        "- GPT -> Generative Pre-trained Transformer\n"
        "- ASL -> American Sign Language\n"
        "- SM -> So much\n"
        "- FLA -> Florida\n"
        "- Abbreviations of every state / country / location / app (like FB -> FaceBook, YT -> YouTube, etc.) / technologies (JS -> JavaScript, TF -> TensorFlow, etc.)\n"
        "Names (e.g., GIANG, JOHN) and acronyms should be preserved or expanded if clear."
        "Only output a sentence or short paragraph, without additional explanation or creative guessing.\n"
        "Only out put the constructed sentence, not the original sentence provided."
        "Make sentences grammatically correct from the keywords / spellings provided. Make it reasonable."
        f"You are given a space-separated string of letters from a fingerspelling segment in ASL video: {text}. "
        "Reconstruct the most likely English sentence using these letters. Prioritize names, abbreviations, and realistic phrases. "
        "Return ONLY the reconstructed sentence or sentences."
    )

    # Generate content using Gemini
    response = model.generate_content(prompt)
    
    return response.text.strip()

if __name__ == '__main__':
    # Example test cases
    samples = ["I L Y S M", "C S", "G I A N G"]
    for asl_text in samples:
        result = analyze_asl_gemini(asl_text)
        print(f"\nASL Text: {asl_text}")
        print(f"Analysis: {result}")
