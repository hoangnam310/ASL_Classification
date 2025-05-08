import pickle
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import time
import threading
import sys
import traceback
import os
from importlib.util import find_spec

# Set up better error handling
def custom_excepthook(exc_type, exc_value, exc_traceback):
    print("Exception detected:")
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("Please check the error above.")

sys.excepthook = custom_excepthook

# Check and create stubs for missing modules if needed
def ensure_module_exists(module_name, stub_content):
    if not os.path.exists(f"{module_name}.py"):
        print(f"Warning: {module_name}.py not found. Creating stub...")
        with open(f"{module_name}.py", 'w') as f:
            f.write(stub_content)
        print(f"Created {module_name}.py stub file.")

# Create stubs for the analysis modules if they don't exist
ensure_module_exists("analyze_nebius", 
                    """def analyze_asl_nebius(text):
    print(f"Analyzing with stub Nebius function: {text}")
    return f"Stub Nebius: {text}"
""")

ensure_module_exists("analyze_gemini", 
                    """def analyze_asl_gemini(text):
    print(f"Analyzing with stub Gemini function: {text}")
    return f"Stub Gemini: {text}"
""")

# Now import the modules (either real or stubbed)
try:
    from analyze_nebius import analyze_asl_nebius
    from analyze_gemini import analyze_asl_gemini
except ImportError as e:
    print(f"Failed to import analysis modules: {e}")
    sys.exit(1)

class HandGestureCNN(nn.Module):
    def __init__(self, num_classes):
        super(HandGestureCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 2), padding=(1, 0))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1, 0))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))
        self.fc_input_size = 128 * 5 * 1
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, self.fc_input_size)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def try_open_camera():
    """Try different methods to open the camera and return a working VideoCapture object"""
    # First try: Default with camera index 0
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print("Camera opened successfully with default backend")
            return cap
        cap.release()
    
    # # Second try: DirectShow (Windows)
    # try:
    #     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    #     if cap.isOpened():
    #         ret, frame = cap.read()
    #         if ret:
    #             print("Camera opened successfully with DirectShow")
    #             return cap
    #         cap.release()
    # except:
    #     pass
    
    # Third try: Microsoft Media Foundation (Windows)
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("Camera opened successfully with MSMF")
                return cap
            cap.release()
    except:
        pass
    
    # Fourth try: Try camera index 1
    cap = cv2.VideoCapture(1)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print("Camera index 1 opened successfully")
            return cap
        cap.release()
    
    print("All camera opening methods failed!")
    return None

def main():
    # Check if model directory exists
    model_dir = "models"
    if not os.path.exists(model_dir):
        print(f"ERROR: Model directory '{model_dir}' not found!")
        print(f"Current working directory: {os.getcwd()}")
        print("Please make sure the models directory exists with the required files.")
        return
    
    # Check if model files exist
    pickle_file = os.path.join(model_dir, "label_map_alphabet_both.pickle")
    model_file = os.path.join(model_dir, "best_cnn_model_alphabet_both.pth")
    
    if not os.path.exists(pickle_file):
        print(f"ERROR: Missing file: {pickle_file}")
        return
        
    if not os.path.exists(model_file):
        print(f"ERROR: Missing file: {model_file}")
        return
    
    print("Loading label mappings...")
    try:
        with open(pickle_file, 'rb') as f:
            label_info = pickle.load(f)
            label_map = label_info['label_map_alphabet']
            reverse_label_map = label_info['reverse_label_map_alphabet']
    except Exception as e:
        print(f"Error loading label mappings: {e}")
        return
    
    print("Loading model...")
    try:
        model = HandGestureCNN(len(label_map))
        model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("Opening camera...")
    cap = try_open_camera()
    if cap is None:
        print("Failed to open camera. Please check your camera connection and permissions.")
        return
    
    print("Initializing MediaPipe...")
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=False, 
                          min_detection_confidence=0.5, 
                          min_tracking_confidence=0.5, 
                          max_num_hands=1)
    
    # Initialize variables for ASL recognition
    sentence = ""
    sentence_log = []
    current_letter = None
    candidate_letter = None
    letter_hold_start = None
    cooldown_time = 1.0
    no_hand_count = 0
    pause_threshold = 15
    cooldown_active = False
    cooldown_start = 0
    required_hold_time = 0.5
    stable_threshold = 5
    letter_history = []
    frame_counter = 0
    message_text = ""
    message_until = 0
    
    print("\nASL Recognition System Started!")
    print("Controls:")
    print("  Press 'a' to analyze/translate the current sentence")
    print("  Press 'q' or ESC to quit")
    print("  Hold hand steady to register a letter")
    print("  Remove hand from view to add a space")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from camera!")
                # Try to reconnect
                cap.release()
                cap = try_open_camera()
                if cap is None:
                    print("Failed to reconnect to camera. Exiting.")
                    break
                continue
    
            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
    
            frame_counter += 1
            if cooldown_active and (time.time() - cooldown_start) > cooldown_time:
                cooldown_active = False
    
            if results.multi_hand_landmarks:
                no_hand_count = 0
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                        mp_drawing_styles.get_default_hand_landmarks_style(),
                                        mp_drawing_styles.get_default_hand_connections_style())
    
                # Extract and normalize hand landmark coordinates
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
    
                data_aux = []
                for lm in hand_landmarks.landmark:
                    data_aux.extend([
                        (lm.x - x_min) / (x_max - x_min) if x_max > x_min else 0,
                        (lm.y - y_min) / (y_max - y_min) if y_max > y_min else 0
                    ])
    
                # Predict the letter using the model
                input_tensor = torch.FloatTensor(np.array(data_aux).reshape(1, 1, 21, 2))
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    predicted_class = predicted.item()
                    confidence_value = confidence.item()
                    prediction = reverse_label_map[predicted_class]
    
                    # Draw bounding box and prediction on frame
                    x1 = max(0, int(x_min * W) - 10)
                    y1 = max(0, int(y_min * H) - 10)
                    x2 = min(W, int(x_max * W) + 10)
                    y2 = min(H, int(y_max * H) + 10)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"{prediction} ({confidence_value:.2f})"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    text_x = x1
                    text_y = y1 - 10 if y1 - 10 > text_size[1] else y1 + text_size[1] + 10
                    cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0], text_y + 5), (0, 255, 0), -1)
                    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    
                # Add prediction to history and check for stable letter
                letter_history.append(prediction)
                if len(letter_history) > stable_threshold:
                    letter_history = letter_history[-stable_threshold:]
    
                most_common = max(set(letter_history), key=letter_history.count)
                if letter_history.count(most_common) >= int(0.8 * stable_threshold):
                    if candidate_letter != most_common:
                        candidate_letter = most_common
                        letter_hold_start = time.time()
                    else:
                        held_time = time.time() - letter_hold_start
                        if held_time >= required_hold_time and not cooldown_active:
                            if current_letter != most_common:
                                sentence += most_common
                                current_letter = most_common
                                cooldown_active = True
                                cooldown_start = time.time()
            else:
                # No hand detected
                no_hand_count += 1
                if no_hand_count >= pause_threshold and len(sentence) > 0 and sentence[-1] != " ":
                    message_text = "Word break detected!"
                    message_until = time.time() + 2.5
                    sentence += " "
                    current_letter = None
                    candidate_letter = None
    
            # Display text on frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text_color = (255, 200, 100)
            bg_color = (30, 30, 30)
    
            # Display current sentence
            sentence_text = f"Sentence: {sentence.strip()}"
            (text_width, text_height), _ = cv2.getTextSize(sentence_text, font, font_scale, font_thickness)
            x, y = 10, H - 30
            cv2.rectangle(frame, (x - 5, y - text_height - 10), (x + text_width + 5, y + 5), bg_color, -1)
            cv2.putText(frame, sentence_text, (x, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
            # Display messages
            if message_text and time.time() < message_until:
                msg_color = (0, 255, 255)  # Bright yellow
                (mw, mh), _ = cv2.getTextSize(message_text, font, 1.0, 2)
                mx, my = 50, 50
                cv2.rectangle(frame, (mx - 10, my - mh - 10), (mx + mw + 10, my + 10), (30, 30, 30), -1)
                cv2.putText(frame, message_text, (mx, my), font, 1.0, msg_color, 2, cv2.LINE_AA)
    
            # Display sentence log (translated sentences)
            for idx, logged_sentence in enumerate(reversed(sentence_log)):
                log_y = 50 + idx * 40
                (lw, lh), _ = cv2.getTextSize(logged_sentence, font, 0.8, 2)
                lx = W - lw - 20
                ly = log_y - lh
                cv2.rectangle(frame, (lx - 10, ly - 10), (lx + lw + 10, log_y + 10), (50, 50, 50), -1)  # Background
                cv2.putText(frame, logged_sentence, (lx, log_y), font, 0.8, (255, 255, 180), 2, cv2.LINE_AA)
    
            # Show the frame
            cv2.imshow('Live ASL Recognition', frame)
            key = cv2.waitKey(1) & 0xFF
            
            # Function to analyze text in a separate thread
            def analyze_and_log(text):
                try:
                    # Try using Gemini first, fall back to Nebius if it fails
                    try:
                        analysis = analyze_asl_gemini(text)
                    except Exception as e:
                        print(f"Gemini analysis failed: {e}")
                        analysis = analyze_asl_nebius(text)
                        
                    print(f"Analysis result: {analysis}")
                    sentence_log.append(f"{analysis}")
                    if len(sentence_log) > 3:
                        sentence_log.pop(0)
                except Exception as e:
                    print(f"Error in analysis: {e}")
                    sentence_log.append(f"Analysis error: {text}")
    
            # Handle key presses
            if key == ord('a'):
                if sentence.strip():
                    message_text = "Analyzing sentence..."
                    message_until = time.time() + 1.5
                    threading.Thread(target=analyze_and_log, args=(sentence.strip(),)).start()
                    sentence = ""
                    current_letter = None
                    candidate_letter = None
                    letter_history.clear()
    
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
    
    except Exception as e:
        print(f"Error in main loop: {e}")
        traceback.print_exc()
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("ASL Recognition system stopped.")

if __name__ == "__main__":
    main()