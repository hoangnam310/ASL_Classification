import os
from openai import OpenAI
from dotenv import load_dotenv
import json

def analyze_asl_nebius(text):
    """
    Analyze ASL space-separated letters and convert to meaningful English sentences
    
    Args:
        text: Space-separated ASL letters/characters
    
    Returns:
        Analyzed text as complete sentences
    """
    # Load environment variables
    load_dotenv()
    
    # Initialize OpenAI client with Nebius settings
    client = OpenAI(
        base_url="https://api.studio.nebius.com/v1/",
        api_key=os.environ.get("NEBIUS_API_KEY"),
    )

    # Create messages for the LLM
    MESSAGES = [
    {
      "role": "system",
      "content": (
        "You are an expert at accurately reconstructing English sentences from ASL fingerspelled letters. "
        "Your job is to return only the most likely sentence(s) the signer intended, based on the detected letter sequence. "
        "Do NOT be creative or infer unrelated phrases. "
        "You may only expand common abbreviations and acronyms into full words. "
        "Here are some common examples:\n"
        "- ILU -> I love you\n"
        "- ILY -> I love you\n"
        "- SW -> Software\n"
        "- SWE -> Software Engineer\n"
        "- CS -> Computer Science\n"
        "- AI -> Artificial Intelligence\n"
        "- USF -> University of South Florida\n"
        "- NYC -> New York City\n"
        "- GPT -> Generative Pre-trained Transformer\n"
        "- ASL -> American Sign Language\n"
        "- SM -> So much\n"
        "- FLA -> Florida\n"
        "- JB -> Job\b"
        "- Abbreviations of every state / country / location / app (like FB -> FaceBook, YT -> YouTube, etc.) / technologies (JS -> JavaScript, TF -> TensorFlow, etc.)\n"
        "Names (e.g., GIANG, JOHN) and acronyms should be preserved or expanded if clear."
        "Only output a sentence or short paragraph, without additional explanation or creative guessing.\n"
        "Only out put the constructed sentence, not the original sentence provided."
        "Make sentences grammatically correct from the keywords / spellings provided. Make it reasonable."
      )
    },
    {
      "role": "user",
      "content": (
        f"You are given a space-separated string of letters from a fingerspelling segment in ASL video: {text}. "
        "Reconstruct the most likely English sentence using these letters. Prioritize names, abbreviations, and realistic phrases. "
        "Return ONLY the reconstructed sentence or sentences."
      )
    }
  ]

    # Generate completion
    completion = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        messages=MESSAGES,
        temperature=0.6
    )

    # Extract response
    response_json = completion.to_json()

    if isinstance(response_json, str):
        response_json = json.loads(response_json)

    analysis = response_json['choices'][0]['message']['content']
    
    return analysis

# ================================================================================

# def main():
#     """
#     Main function to read ASL recognition results and analyze them
#     """
#     # Path to the results JSON file
#     results_path = "output_folder/text_result/asl_recognition_results.json"
    
#     try:
#         # Read the JSON file
#         with open(results_path, 'r') as f:
#             data = json.load(f)
        
#         # Check if there are any results
#         if not data.get("results"):
#             print("No results found in the JSON file.")
#             return
        
#         # Process each video result
#         for i, result in enumerate(data["results"]):
#             print(f"\nVideo {i+1}:")
#             print(f"Original detected sentence: {result['detected_sentence']}")
#             print(f"Space separated letters: {result['space_separated_letters']}")
            
#             # Analyze the space separated letters
#             # analysis = analyze_asl(result['detected_sentence'])
#             analysis = analyze_asl("Me SW agent")
            
#             print("\nAnalysis:")
#             print(analysis)
            
#             # Save analysis to result dictionary
#             result["analysis"] = analysis
        
#         # Save updated results back to the file
#         with open(results_path, 'w') as f:
#             json.dump(data, f, indent=2)
            
#         print(f"\nAnalysis results saved to {results_path}")
            
#     except FileNotFoundError:
#         print(f"Error: Could not find results file at {results_path}")
#     except json.JSONDecodeError:
#         print(f"Error: The file at {results_path} is not valid JSON")
#     except Exception as e:
#         print(f"Error: An unexpected error occurred: {e}")

# if __name__ == "__main__":
#     main()