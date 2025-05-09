import pickle
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import time
from collections import deque
import threading
import os
from analyze_gemini import analyze_asl_gemini

MODEL_SAVE_DIR = './models' 
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_lstm_model_sequences_sorted.pth')
LABEL_MAP_PATH = os.path.join(MODEL_SAVE_DIR, 'label_map_sequences.pickle')

SEQUENCE_LENGTH = 10
NUM_LANDMARKS = 21
FEATURES_PER_LANDMARK = 2 # 2 if 2 dimension
FEATURES_PER_HAND = NUM_LANDMARKS * FEATURES_PER_LANDMARK # 42
TARGET_FEATURES_PER_FRAME = FEATURES_PER_HAND * 2         # 84 
PREDICTION_THRESHOLD = 0.7  
PAUSE_THRESHOLD_FRAMES = 10   # Frames without hands to trigger word break
WINDOW_NAME = 'Live ASL Recognition (LSTM)'

stable_threshold = 8  # Frames needed for non-J/Z letter stability
required_hold_time = 0.0  # Min time a stable non-J/Z letter needs to be held (can be 0)
cooldown_time = 1.5  # Min time before *any* letter (incl. J/Z) can be added again

# --- Define LSTM Model Class (Must match training architecture) ---
class HandGestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.5):
        super(HandGestureLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        lstm_dropout = dropout_prob if num_layers > 1 else 0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=lstm_dropout)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Detach states to prevent backprop through time if not needed
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        # We only need the output of the last time step
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
if not os.path.exists(LABEL_MAP_PATH):
     print(f"Error: Label map file not found at {LABEL_MAP_PATH}.")
     exit()
try:
    with open(LABEL_MAP_PATH, 'rb') as f:
        label_info = pickle.load(f)
        # Ensure the expected keys exist
        if 'label_map' not in label_info or 'reverse_label_map' not in label_info:
             raise ValueError("Label map file missing 'label_map' or 'reverse_label_map' key.")
        label_map = label_info['label_map']
        reverse_label_map = label_info['reverse_label_map']
        num_classes = len(label_map)
        print(f"Loaded {num_classes} classes.")
except Exception as e:
    print(f"Error loading label map file: {e}")
    exit()

# --- Load Model ---
print(f"Loading model from: {BEST_MODEL_PATH}")
if not os.path.exists(BEST_MODEL_PATH):
    print(f"Error: Model file not found at {BEST_MODEL_PATH}.")
    exit()

input_size = TARGET_FEATURES_PER_FRAME # 84
hidden_size = 128 
num_layers = 2
dropout_prob = 0.5

model = HandGestureLSTM(input_size, hidden_size, num_layers, num_classes, dropout_prob)
try:
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model state dictionary: {e}")
    if "size mismatch" in str(e):
        print("This often means the model architecture (input_size, hidden_size, num_layers, num_classes) defined here")
        print("does not match the architecture of the saved model file.")
    exit()


# --- MediaPipe Initialization ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,    
    max_num_hands=2,              
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5   
)

# --- Helper Function for Hand Processing ---
def process_hand_landmarks(landmarks, image_shape):
    """
    Extracts and normalizes hand landmarks relative to the hand's bounding box.
    Returns a dictionary containing normalized features and bounding box coords, or None.
    """
    if not landmarks: return None

    image_height, image_width = image_shape[:2]

    # Get absolute pixel coordinates
    x_coords = [lm.x * image_width for lm in landmarks.landmark]
    y_coords = [lm.y * image_height for lm in landmarks.landmark]

    # Calculate bounding box
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    box_width = x_max - x_min
    box_height = y_max - y_min

    # Avoid division by zero if bounding box is degenerate
    if box_width == 0 or box_height == 0: return None

    # Normalize landmarks relative to the bounding box top-left corner
    normalized_features = []
    for lm in landmarks.landmark:
        # Calculate position relative to top-left corner of the box
        relative_x = lm.x * image_width - x_min
        relative_y = lm.y * image_height - y_min
        # Normalize by box dimensions
        norm_x = relative_x / box_width
        norm_y = relative_y / box_height
        normalized_features.extend([norm_x, norm_y])

    if len(normalized_features) != FEATURES_PER_HAND:
         print(f"Warning: Expected {FEATURES_PER_HAND} features, got {len(normalized_features)}. Padding/truncating.")
         normalized_features = normalized_features[:FEATURES_PER_HAND] # Truncate
         while len(normalized_features) < FEATURES_PER_HAND: # Pad
             normalized_features.append(0.0)


    return {
        'features': normalized_features,
        'x_min': x_min, 'y_min': y_min,
        'x_max': x_max, 'y_max': y_max
    }

# --- Live Test Initialization ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Use 0 for default webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Deques for managing sequences and stability
sequence_buffer = deque(maxlen=SEQUENCE_LENGTH) # Stores feature vectors for recent frames
letter_history = deque(maxlen=stable_threshold) # Stores recent non-J/Z predictions for stability check

# State variables
sentence = ""               
sentence_log = []
no_hand_count = 0
message_text = ""
message_until = 0
current_letter = None
candidate_letter = None
letter_hold_start = 0
cooldown_active = False
cooldown_start = 0

# FPS calculation
frame_count = 0
start_time = time.time()

print("\n--- Starting Live Recognition ---")
print(f"Model: {BEST_MODEL_PATH}")
print(f"Confidence Threshold: {PREDICTION_THRESHOLD}")
print(f"Stability Threshold (frames): {stable_threshold}")
print(f"Cooldown Time (seconds): {cooldown_time}")
print("Press 'A' to analyze current sentence with Gemini.")
print("Press 'Q' or 'Esc' to exit.")
print("------------------------------------")

# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame. Exiting.")
        break

    frame_time = time.time() # Timestamp for this frame
    frame_count += 1
    H, W, _ = frame.shape
    display_frame = frame.copy() # Create a copy for drawing annotations

    # --- 2. MediaPipe Hand Tracking ---
    # Convert frame to RGB (MediaPipe expects RGB)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Improve performance by marking image as not writeable before processing
    image_rgb.flags.writeable = False
    results = hands.process(image_rgb)
    # Mark image as writeable again for drawing
    image_rgb.flags.writeable = True

    # --- 3. Cooldown Check ---
    if cooldown_active and frame_time - cooldown_start >= cooldown_time:
        cooldown_active = False
        candidate_letter = None # Clear candidate when cooldown ends

    # --- 4. Landmark Extraction and Feature Processing ---
    frame_features = np.zeros(TARGET_FEATURES_PER_FRAME, dtype=np.float32) # Initialize empty features
    hands_detected_this_frame = bool(results.multi_hand_landmarks)

    if hands_detected_this_frame:
        no_hand_count = 0 # Reset no-hand counter
        processed_hands = []

        # Process landmarks for up to 2 detected hands
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
            # Draw landmarks and connections on the display frame
            mp_drawing.draw_landmarks(
                display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Extract and normalize features for this hand
            hand_data = process_hand_landmarks(hand_landmarks, (H, W))
            if hand_data:
                processed_hands.append(hand_data)
                # Draw bounding box around the hand
                x1, y1 = int(hand_data['x_min']) - 10, int(hand_data['y_min']) - 10
                x2, y2 = int(hand_data['x_max']) + 10, int(hand_data['y_max']) + 10
                box_color = (0, 255, 0) if hand_idx == 0 else (0, 200, 255) # Green for 1st, Yellow for 2nd
                cv2.rectangle(display_frame, (max(0, x1), max(0, y1)), (min(W, x2), min(H, y2)), box_color, 2)

        # Populate the frame_features array
        if len(processed_hands) >= 1:
            features1 = processed_hands[0]['features']
            len1 = min(len(features1), FEATURES_PER_HAND)
            frame_features[:len1] = features1[:len1]
            if len(processed_hands) >= 2:
                features2 = processed_hands[1]['features']
                len2 = min(len(features2), FEATURES_PER_HAND)
                frame_features[FEATURES_PER_HAND : FEATURES_PER_HAND + len2] = features2[:len2]
    else:
        # No hands detected in this frame
        no_hand_count += 1
        # Reset stability tracking immediately when hands disappear
        letter_history.clear()
        candidate_letter = None

    # Add the processed features (or zeros if no hands) to the sequence buffer
    sequence_buffer.append(frame_features)

    # --- 5. Prediction ---
    prediction = None # Prediction variable for the STABILITY check
    current_prediction = None # Prediction from this frame's inference

    # Only predict if sequence buffer is full AND hands were detected
    if hands_detected_this_frame and len(sequence_buffer) == SEQUENCE_LENGTH:
        try:
            # Prepare input tensor for the model
            input_sequence = np.array(list(sequence_buffer), dtype=np.float32)
            # Add batch dimension (batch_size=1)
            input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(device)

            # Perform inference
            with torch.inference_mode(): # More efficient than torch.no_grad() for inference
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                # Get the top prediction and its confidence
                confidence, predicted_idx = torch.max(probabilities, 1)
                pred_idx = predicted_idx.item()
                conf_val = confidence.item()

                # Check if confidence meets the threshold
                if conf_val >= PREDICTION_THRESHOLD:
                    if pred_idx in reverse_label_map:
                        # Get the predicted letter string
                        current_prediction = reverse_label_map[pred_idx]
                        current_confidence = conf_val

                        # --- Draw Prediction Text (Moved Earlier) ---
                        # Display the current prediction regardless of type (J/Z or other)
                        try:
                            first_hand = results.multi_hand_landmarks[0]
                            # Calculate position near the top of the first hand
                            x_coords = [lm.x for lm in first_hand.landmark]
                            y_coords = [lm.y for lm in first_hand.landmark]
                            x_min, y_min = min(x_coords), min(y_coords)
                            x1_text = max(0, int(x_min * W) - 10)
                            y1_text = max(0, int(y_min * H) - 10)

                            text_to_display = f"{current_prediction} ({current_confidence:.2f})"
                            text_size, _ = cv2.getTextSize(text_to_display, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                            # Position text above the hand box
                            text_x = x1_text
                            text_y = y1_text - 10 if y1_text - 10 > text_size[1] else y1_text + text_size[1] + 30

                            # Draw background rectangle for text
                            cv2.rectangle(display_frame, (text_x, text_y - text_size[1] - 5),
                                          (text_x + text_size[0], text_y + 5), (0, 255, 0), -1) # Green background
                            # Draw prediction text
                            cv2.putText(display_frame, text_to_display, (text_x, text_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA) # Black text
                        except IndexError:
                             print("Warning: Hand landmarks disappeared unexpectedly during drawing.")
                        # --- End of Moved Drawing Block ---


                        # --- Handle J/Z immediate addition OR set prediction for stability ---
                        if current_prediction in ['J']:
                            # Try to add J/Z immediately if cooldown allows and not duplicate
                            if not cooldown_active and (not sentence or sentence[-1] != current_prediction):
                                sentence += current_prediction
                                current_letter = current_prediction # Update last added
                                cooldown_active = True
                                cooldown_start = frame_time
                                print(f"Added (Motion): {current_prediction} | Sentence: {sentence}")
                                # Reset stability mechanisms after adding J/Z
                                letter_history.clear()
                                candidate_letter = None
                            # Set 'prediction' to None for J/Z cases to bypass stability check below
                            prediction = None
                        else:
                            # It's not J or Z, assign it to 'prediction' for the stability check
                            prediction = current_prediction
                            # We already have current_confidence if needed later

        except Exception as e:
            print(f"Prediction error: {e}")
            prediction = None # Ensure prediction is None on error


    # --- 6. Letter Stability Check (for non-J/Z) ---
    # This block only runs if 'prediction' is not None (i.e., a non-J/Z letter was predicted)
    if prediction:
        letter_history.append(prediction) # Add the potential stable letter to history

        # Check if the history buffer is full enough to check stability
        if len(letter_history) >= stable_threshold:
            # Find the most frequent letter in the recent history
            try:
                 most_common = max(set(letter_history), key=list(letter_history).count)
                 # Check if it's consistently the most common
                 # (e.g., >= 85% of the frames in the stable_threshold window)
                 if list(letter_history).count(most_common) >= int(0.85 * stable_threshold):
                     # We have a stable candidate
                     if candidate_letter != most_common:
                         # New stable candidate detected
                         candidate_letter = most_common
                         letter_hold_start = frame_time # Start timer for hold duration
                     else:
                         # Candidate is still the same, check hold time
                         held_time = frame_time - letter_hold_start
                         if held_time >= required_hold_time and not cooldown_active:
                             # Check if it's different from the last *added* letter
                             if not sentence or sentence[-1] != most_common:
                                 # Add the stable letter to the sentence!
                                 sentence += most_common
                                 current_letter = most_common # Update last added
                                 cooldown_active = True
                                 cooldown_start = frame_time
                                 print(f"Added: {most_common} | Sentence: {sentence}")
                                 # Clear history and candidate after successful addition
                                 letter_history.clear()
                                 candidate_letter = None
            except ValueError: # Handles case where letter_history might be empty unexpectedly
                 pass


    # --- 7. Word Break Detection ---
    # Check if hands have been absent for enough frames and sentence isn't empty/ending with space
    if no_hand_count >= PAUSE_THRESHOLD_FRAMES and sentence and not sentence.endswith(" "):
        message_text = "Word break!"
        message_until = frame_time + 2.0 # Show message for 2 seconds
        sentence += " "
        print("Added: [SPACE]")
        # Reset states after adding space
        current_letter = None
        candidate_letter = None
        letter_history.clear()
        # prediction_buffer.clear() # Also clear prediction buffer if using it

    # --- 8. Drawing Annotations ---
    font = cv2.FONT_HERSHEY_SIMPLEX

    # FPS Display (Top Right)
    fps = frame_count / (frame_time - start_time) if (frame_time - start_time) > 0 else 0
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (W - 100, 30), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    # Hand Count Display (Top Left)
    hands_text = f"Hands: {len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0}"
    cv2.putText(display_frame, hands_text, (10, 30), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA) # Red text

    # Candidate Letter Status (Top Left)
    if candidate_letter:
        held_time = frame_time - letter_hold_start
        hold_status = f"Candidate: {candidate_letter} ({held_time:.1f}s)"
        cv2.putText(display_frame, hold_status, (10, 60), font, 0.7, (255, 255, 0), 2, cv2.LINE_AA) # Cyan text

    # Cooldown Status (Top Left)
    if cooldown_active:
        remaining = max(0, cooldown_time - (frame_time - cooldown_start))
        cooldown_text = f"Cooldown: {remaining:.1f}s"
        cv2.putText(display_frame, cooldown_text, (10, 90), font, 0.7, (0, 165, 255), 2, cv2.LINE_AA) # Orange text

    # Current Sentence Display (Bottom Left)
    sentence_display = f"Sentence: {sentence}"
    (tw, th), _ = cv2.getTextSize(sentence_display, font, 0.9, 2)
    sx, sy = 10, H - 20 # Position near bottom left
    # Background rectangle for sentence
    cv2.rectangle(display_frame, (sx - 5, sy - th - 10), (sx + tw + 5, sy + 5), (50, 50, 50), -1) # Dark grey bg
    cv2.putText(display_frame, sentence_display, (sx, sy), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA) # White text

    # Temporary Messages Display (Middle Top)
    if message_text and frame_time < message_until:
        (mw, mh), _ = cv2.getTextSize(message_text, font, 1.0, 2)
        mx = (W - mw) // 2 # Center horizontally
        my = 60           # Position below FPS/Hand count
        # Background rectangle for message
        cv2.rectangle(display_frame, (mx - 10, my - mh - 10), (mx + mw + 10, my + 10), (0, 0, 0), -1) # Black bg
        cv2.putText(display_frame, message_text, (mx, my), font, 1.0, (0, 255, 255), 2, cv2.LINE_AA) # Yellow text
    elif frame_time >= message_until:
        message_text = "" # Clear message after duration

    # API Sentence Log Display 
    log_font_scale = 1.6
    log_thickness = 5
    log_color = (135, 206, 235) 
    log_bg_color = (80, 80, 80) 
    log_y_start = 70 
    log_line_height = 35 
    max_log_lines = 3
    log_line_spacing = 10
    # Display from oldest to newest (no reversed), top to bottom
    for idx, logged_sentence in enumerate(sentence_log[-max_log_lines:]):
        (lw, lh), _ = cv2.getTextSize(logged_sentence, font, log_font_scale, log_thickness)
        
        log_y = log_y_start + idx * (lh + log_line_spacing)  # Ensure proper spacing using text height
        lx = W - lw - 20  # Right-aligned

        # Draw background rectangle with padding
        cv2.rectangle(display_frame, (lx - 8, log_y - lh - 5), (W - 10, log_y + 5), log_bg_color, -1)

        # Draw the log text
        cv2.putText(display_frame, logged_sentence, (lx, log_y), font, log_font_scale, log_color, log_thickness, cv2.LINE_AA)



    # --- 9. Display the Frame ---
    cv2.imshow(WINDOW_NAME, display_frame)

    # --- 10. Handle User Input ---
    key = cv2.waitKey(1) & 0xFF # Wait 1ms for key press

    # Analyze sentence with API ('a' key)
    if key == ord('a'):
        trimmed_sentence = sentence.strip()
        if trimmed_sentence:
            print(f"\nSending to API: '{trimmed_sentence}'")
            message_text = "Analyzing..."
            message_until = frame_time + cooldown_time + 1.5 # Show analysis message

            # Define the API call function to run in a thread
            def analyze_and_log(text_to_analyze):
                global sentence_log # Allow modification of global log list
                try:
                    # Replace with your actual API call
                    analysis = analyze_asl_gemini(text_to_analyze)
                    print(f"API Response: {analysis}")
                    sentence_log.append(f"{analysis}") # Add result to log
                except Exception as api_e:
                    print(f"Error during API call: {api_e}")
                    sentence_log.append("API Error") # Log error
                finally:
                    # Keep only the last N logs
                    if len(sentence_log) > max_log_lines:
                        sentence_log = sentence_log[-max_log_lines:]

            # Start API call in a separate thread to avoid freezing the GUI
            threading.Thread(target=analyze_and_log, args=(trimmed_sentence,), daemon=True).start()

            # Reset sentence and related states
            sentence = ""
            current_letter = None
            candidate_letter = None
            letter_history.clear()
            cooldown_active = False # Reset cooldown as sentence is cleared
        else:
            print("Sentence is empty, not sending to API.")
            message_text = "Sentence empty!"
            message_until = frame_time + 2.0

    # Quit ('q' or Esc key)
    if key == ord('q') or key == 27: # 27 is the Esc key code
        print("\nExiting...")
        break

# --- 11. Cleanup ---
cap.release() # Release the webcam
cv2.destroyAllWindows() # Close all OpenCV windows
# Add a small delay to ensure windows are closed properly, especially on some OS
for i in range(5): cv2.waitKey(1)
if 'hands' in locals() and hands:
    hands.close() # Close the MediaPipe hands instance
print("Resources released.")