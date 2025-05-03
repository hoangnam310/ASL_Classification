import pickle
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import time
from collections import deque
import threading
import os # Added for path joining

# --- Assumed API Functions (Make sure these files exist) ---
# Create dummy functions if the real ones aren't available yet
try:
    from analyze_gemini import analyze_asl_gemini
except ImportError:
    print("Warning: analyze_gemini not found. Using dummy function.")
    def analyze_asl_gemini(text):
        print(f"DUMMY: Analyzing '{text}' with Gemini...")
        time.sleep(1) # Simulate API call
        return f"Gemini: '{text.upper()}'"

# --- Configuration ---
MODEL_SAVE_DIR = './models'
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_lstm_model_sequences.pth')
LABEL_MAP_PATH = os.path.join(MODEL_SAVE_DIR, 'label_map_sequences.pickle')

SEQUENCE_LENGTH = 10          # Must match the training sequence length
NUM_LANDMARKS = 21
FEATURES_PER_LANDMARK = 2
FEATURES_PER_HAND = NUM_LANDMARKS * FEATURES_PER_LANDMARK # 42
TARGET_FEATURES_PER_FRAME = FEATURES_PER_HAND * 2         # 84 (for two hands, padded)
PREDICTION_THRESHOLD = 0.6   # Minimum confidence to display prediction
PREDICTION_BUFFER_SIZE = 3    # Require prediction to be stable for N consecutive sequences
PAUSE_THRESHOLD_FRAMES = 10   # Frames without hands to trigger word break
WINDOW_NAME = 'Live ASL Recognition (LSTM)'

# --- Added for single-frame approach ---
stable_threshold = 8  # Frames needed for letter stability
required_hold_time = 0.5  # Time letter needs to be held
cooldown_time = 1.5  # Time before same letter can be added again

# --- Define LSTM Model Class (Must match training architecture) ---
class HandGestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.5):
        super(HandGestureLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# --- Device Setup ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# --- Load Label Mapping ---
print(f"Loading label mapping from: {LABEL_MAP_PATH}")
try:
    with open(LABEL_MAP_PATH, 'rb') as f:
        label_info = pickle.load(f)
        label_map = label_info['label_map']
        reverse_label_map = label_info['reverse_label_map']
        num_classes = len(label_map)
        print(f"Loaded {num_classes} classes.")
except FileNotFoundError:
    print(f"Error: Label map file not found at {LABEL_MAP_PATH}.")
    exit()
except Exception as e:
    print(f"Error loading label map file: {e}")
    exit()

# --- Load Model ---
print(f"Loading model from: {BEST_MODEL_PATH}")
# Determine model parameters (these should ideally be saved with the model or known)
input_size = TARGET_FEATURES_PER_FRAME # 84
hidden_size = 128 # Must match the hidden_size used during training
num_layers = 2    # Must match the num_layers used during training
dropout_prob = 0.5 # Must match dropout used during training

model = HandGestureLSTM(input_size, hidden_size, num_layers, num_classes, dropout_prob)
try:
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval() # Set model to evaluation mode
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {BEST_MODEL_PATH}.")
    exit()
except Exception as e:
    print(f"Error loading model state dictionary: {e}")
    exit()


# --- MediaPipe Initialization ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2, # Detect up to two hands
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Helper Function for Hand Processing ---
def process_hand_landmarks(landmarks, image_shape):
    """Normalizes landmarks relative to the hand's bounding box."""
    if not landmarks: return None
    
    # Get coordinate boundaries
    x_coords = [lm.x * image_shape[1] for lm in landmarks.landmark]
    y_coords = [lm.y * image_shape[0] for lm in landmarks.landmark]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Check for zero-size bounding box
    box_width = x_max - x_min
    box_height = y_max - y_min
    if box_width == 0 or box_height == 0: return None
    
    # Create normalized features for the hand
    normalized_features = []
    for lm in landmarks.landmark:
        norm_x = (lm.x * image_shape[1] - x_min) / box_width
        norm_y = (lm.y * image_shape[0] - y_min) / box_height
        normalized_features.extend([norm_x, norm_y])
    
    return {
        'features': normalized_features,
        'x_min': x_min,
        'y_min': y_min, 
        'x_max': x_max,
        'y_max': y_max
    }

# --- Live Test Initialization ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

sequence_buffer = deque(maxlen=SEQUENCE_LENGTH) # Buffer for frame features
prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE) # Buffer for consecutive predictions
sentence = ""
sentence_log = [] # Store API analysis results
no_hand_count = 0
message_text = ""
message_until = 0

# For single frame approach
letter_history = []
current_letter = None
candidate_letter = None
letter_hold_start = time.time()
cooldown_active = False
cooldown_start = time.time()

# For FPS calculation
frame_count = 0
start_time = time.time()

# Visual theme settings
THEME = {
    'font': cv2.FONT_HERSHEY_SIMPLEX,
    'font_scale': 1.0,
    'thickness': 2,
    'text_color': (255, 200, 100),  # Light orange/yellow
    'bg_color': (30, 30, 30),       # Dark gray
    'hand_box_color': (0, 255, 0),  # Green
    'second_hand_color': (0, 200, 255),  # Light blue
    'message_color': (0, 255, 255), # Bright yellow
    'log_color': (255, 255, 180),   # Light yellow
    'log_bg_color': (50, 50, 50),   # Mid gray
    'fps_color': (0, 0, 255)        # Red
}

print("\nStarting Live Recognition...")
print("Press 'A' to analyze current sentence.")
print("Press 'Q' or 'Esc' to exit.")

# --- Main Loop ---
while True:
    # --- Frame Capture and FPS ---
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    H, W, _ = frame.shape
    display_frame = frame.copy() # Work on a copy for drawing

    # --- MediaPipe Processing ---
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False # Optimize
    results = hands.process(image_rgb)
    image_rgb.flags.writeable = True

    # Check if cooldown is over
    if cooldown_active and time.time() - cooldown_start >= cooldown_time:
        cooldown_active = False

    # --- Combined Landmark Extraction and Sequence Management ---
    frame_features = np.zeros(TARGET_FEATURES_PER_FRAME, dtype=np.float32) # Default empty frame
    prediction = None
    confidence_value = 0.0
    
    # Process hands if detected
    if results.multi_hand_landmarks:
        no_hand_count = 0 # Reset counter when hands are detected
        
        # Process up to 2 hands
        processed_hands = []
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):  # Limit to 2 hands
            # Draw landmarks on display frame
            mp_drawing.draw_landmarks(
                display_frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Process hand landmarks
            hand_data = process_hand_landmarks(hand_landmarks, (H, W))
            if hand_data:
                processed_hands.append(hand_data)
                
                # Draw bounding box for this hand
                x1 = max(0, int(hand_data['x_min']) - 10)
                y1 = max(0, int(hand_data['y_min']) - 10)
                x2 = min(W, int(hand_data['x_max']) + 10)
                y2 = min(H, int(hand_data['y_max']) + 10)
                
                # Different color for each hand
                box_color = THEME['hand_box_color'] if hand_idx == 0 else THEME['second_hand_color']
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
        
        # Fill the feature vector with both hands' data
        if len(processed_hands) >= 1:
            # First hand features fill the first half
            hand1_features = processed_hands[0]['features']
            features_len = min(len(hand1_features), FEATURES_PER_HAND)
            frame_features[:features_len] = hand1_features[:features_len]
            
            # If we have a second hand, fill the second half
            if len(processed_hands) >= 2:
                hand2_features = processed_hands[1]['features']
                features_len = min(len(hand2_features), FEATURES_PER_HAND)
                frame_features[FEATURES_PER_HAND:FEATURES_PER_HAND+features_len] = hand2_features[:features_len]
        
        # Add features to sequence buffer
        sequence_buffer.append(frame_features)
        
        # Only make prediction when sequence buffer is full AND hands are detected
        if len(sequence_buffer) == SEQUENCE_LENGTH:
            try:
                # Prepare sequence tensor
                input_sequence = np.array(list(sequence_buffer), dtype=np.float32)
                input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(device)
                
                with torch.inference_mode():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                    pred_idx = predicted_idx.item()
                    conf_val = confidence.item()
                    
                    if conf_val >= PREDICTION_THRESHOLD:
                        if pred_idx in reverse_label_map:
                            prediction = reverse_label_map[pred_idx]
                            confidence_value = conf_val
                            
                            # Display prediction near the first hand if available
                            if results.multi_hand_landmarks:
                                # Get first hand position for text placement
                                first_hand = results.multi_hand_landmarks[0]
                                x_coords = [lm.x for lm in first_hand.landmark]
                                y_coords = [lm.y for lm in first_hand.landmark]
                                x_min, y_min = min(x_coords), min(y_coords)
                                
                                x1 = max(0, int(x_min * W) - 10)
                                y1 = max(0, int(y_min * H) - 10)
                                
                                text = f"{prediction} ({confidence_value:.2f})"
                                text_size = cv2.getTextSize(text, THEME['font'], 1, 2)[0]
                                text_x = x1
                                text_y = y1 - 10 if y1 - 10 > text_size[1] else y1 + text_size[1] + 10
                                cv2.rectangle(display_frame, (text_x, text_y - text_size[1] - 5), 
                                             (text_x + text_size[0], text_y + 5), THEME['hand_box_color'], -1)
                                cv2.putText(display_frame, text, (text_x, text_y), 
                                           THEME['font'], 1, (0, 0, 0), 2, cv2.LINE_AA)
            except Exception as e:
                print(f"Prediction error: {e}")
                prediction = None
        
            # Add prediction to buffer
            prediction_buffer.append(prediction)
        
            # Apply letter stability mechanism
            if prediction:
                letter_history.append(prediction)
                if len(letter_history) > stable_threshold:
                    letter_history = letter_history[-stable_threshold:]
        
                # Check for stable letter
                if len(letter_history) >= stable_threshold:
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
                                    print(f"Added: {most_common} | Sentence: {sentence}")
    else:
        # No hands detected - reset prediction state
        no_hand_count += 1
        candidate_letter = None
        letter_history.clear()
        sequence_buffer.clear()  # Clear the sequence buffer when no hands are present
        prediction_buffer.clear()
    
    # Check for word break
    if no_hand_count >= PAUSE_THRESHOLD_FRAMES and len(sentence) > 0 and sentence[-1] != " ":
        message_text = "Word break detected!"
        message_until = time.time() + 2.0
        sentence += " "
        current_letter = None
        candidate_letter = None
        prediction_buffer.clear()
        print("Added: [SPACE]")

    # --- Drawing ---
    # Display Info Texts
    font = THEME['font']
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (W - 100, 30), font, 0.7, THEME['fps_color'], 2, cv2.LINE_AA)
    
    # Display number of hands detected
    if results.multi_hand_landmarks:
        hands_text = f"Hands: {len(results.multi_hand_landmarks)}"
        cv2.putText(display_frame, hands_text, (10, 30), font, 0.7, (255, 100, 255), 2, cv2.LINE_AA)
    
    # Display candidate letter status
    if candidate_letter:
        held_time = time.time() - letter_hold_start
        hold_status = f"Holding: {candidate_letter} ({held_time:.1f}s)"
        cv2.putText(display_frame, hold_status, (10, 60), font, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    
    # Display cooldown status
    if cooldown_active:
        remaining = cooldown_time - (time.time() - cooldown_start)
        cooldown_text = f"Cooldown: {remaining:.1f}s"
        cv2.putText(display_frame, cooldown_text, (10, 90), font, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

    # Display Sentence
    sentence_text = f"Sentence: {sentence.strip()}"
    (text_width, text_height), _ = cv2.getTextSize(sentence_text, font, THEME['font_scale'], THEME['thickness'])
    sx, sy = 10, H - 30
    cv2.rectangle(display_frame, (sx - 5, sy - text_height - 10), (sx + text_width + 5, sy + 5), THEME['bg_color'], -1)
    cv2.putText(display_frame, sentence_text, (sx, sy), font, THEME['font_scale'], THEME['text_color'], THEME['thickness'], cv2.LINE_AA)

    # Display Temporary Messages (like word break)
    if message_text and time.time() < message_until:
        (mw, mh), _ = cv2.getTextSize(message_text, font, 1.0, 2)
        mx, my = 50, 100 # Position message higher up
        cv2.rectangle(display_frame, (mx - 10, my - mh - 10), (mx + mw + 10, my + 10), THEME['bg_color'], -1)
        cv2.putText(display_frame, message_text, (mx, my), font, 1.0, THEME['message_color'], 2, cv2.LINE_AA)
    elif time.time() >= message_until:
        message_text = "" # Clear message after duration

    # Display sentence log (translated sentences) - Top Right
    log_font_scale = 0.8
    log_thickness = 2
    log_y_start = 80
    log_line_height = 40
    for idx, logged_sentence in enumerate(reversed(sentence_log[-3:])): # Show last 3 logs
        log_y = log_y_start + idx * log_line_height
        (lw, lh), _ = cv2.getTextSize(logged_sentence, font, log_font_scale, log_thickness)
        lx = W - lw - 20
        ly = log_y
        # Draw background rectangle
        cv2.rectangle(display_frame, (lx - 10, ly - lh - 10), 
                      (W - 10, log_y + 10), THEME['log_bg_color'], -1)
        # Draw text
        cv2.putText(display_frame, logged_sentence, (lx, log_y), font, 
                   log_font_scale, THEME['log_color'], log_thickness, cv2.LINE_AA)

    # --- Display Frame ---
    cv2.imshow(WINDOW_NAME, display_frame)

    # --- Handle Keys ---
    key = cv2.waitKey(1) & 0xFF

    # Function to run analysis in a thread
    def analyze_and_log(text_to_analyze):
        # *** Use 'global' to modify the global sentence_log list ***
        global sentence_log
        print(f"API Call: Analyzing '{text_to_analyze}'")
        try:
            # analysis = analyze_asl_nebius(text) # Your Nebius function
            analysis = analyze_asl_gemini(text_to_analyze) # Your Gemini function
            print(f"API Response: {analysis}")
            sentence_log.append(f"{analysis}") # Add result to log
            # Keep only the last 3 logs
            if len(sentence_log) > 3:
                sentence_log = sentence_log[-3:]
        except Exception as api_e:
            print(f"Error during API call: {api_e}")
            sentence_log.append("API Error") # Log error
            if len(sentence_log) > 3:
                sentence_log = sentence_log[-3:]


    if key == ord('a'):
        trimmed_sentence = sentence.strip()
        if trimmed_sentence:
            print(f"Sending to API: '{trimmed_sentence}'")
            # Start analysis in a separate thread to avoid blocking the UI
            threading.Thread(target=analyze_and_log, args=(trimmed_sentence,), daemon=True).start()
            # Reset sentence and related tracking variables
            sentence = ""
            current_letter = None
            candidate_letter = None
            cooldown_active = False
            letter_history.clear()
            message_text = "Sent to API!"
            message_until = time.time() + 2.0
        else:
            print("Sentence is empty, not sending to API.")
            message_text = "Sentence empty!"
            message_until = time.time() + 2.0


    if key == ord('q') or key == 27: # 27 is Escape key
        break

# --- Cleanup ---
print("\nExiting...")
cap.release()
try:
    cv2.destroyWindow(WINDOW_NAME)
except cv2.error: pass
cv2.destroyAllWindows()
for i in range(5): cv2.waitKey(1)
hands.close()
print("Resources released.")