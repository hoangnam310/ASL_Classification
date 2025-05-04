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

# --- Configuration ---
MODEL_SAVE_DIR = './models' # IMPORTANT: Point to the directory containing the 3D-trained model
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_lstm_model_sequences_sorted_1.pth') # Assumed name for 3D model
LABEL_MAP_PATH = os.path.join(MODEL_SAVE_DIR, 'label_map_sequences.pickle') # Assumed name for 3D label map

SEQUENCE_LENGTH = 10
NUM_LANDMARKS = 21
# --- >>> CHANGE 1: Update Feature Constants for 3D <<< ---
USE_Z_COORDINATE = True # Explicitly set for clarity
FEATURES_PER_LANDMARK = 3 # X, Y, Z
FEATURES_PER_HAND = NUM_LANDMARKS * FEATURES_PER_LANDMARK # 21 * 3 = 63
TARGET_FEATURES_PER_FRAME = FEATURES_PER_HAND * 2         # 63 * 2 = 126
# --- <<< End of CHANGE 1 >>> ---

PREDICTION_THRESHOLD = 0.7
PAUSE_THRESHOLD_FRAMES = 10   # Frames without hands to trigger word break
WINDOW_NAME = 'Live ASL Recognition (LSTM - 3D)'

# Stability and cooldown settings (remain the same)
stable_threshold = 8
required_hold_time = 0.0
cooldown_time = 1.5

# --- Define LSTM Model Class (Must match training architecture) ---
# (Model class definition remains the same as provided)
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
if not os.path.exists(LABEL_MAP_PATH):
     print(f"Error: Label map file not found at {LABEL_MAP_PATH}.")
     exit()
try:
    with open(LABEL_MAP_PATH, 'rb') as f:
        label_info = pickle.load(f)
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

# --- >>> CHANGE 2: Update Input Size for Model Initialization <<< ---
input_size = TARGET_FEATURES_PER_FRAME # Should now be 126
# --- <<< End of CHANGE 2 >>> ---
hidden_size = 128
num_layers = 2
dropout_prob = 0.5

model = HandGestureLSTM(input_size, hidden_size, num_layers, num_classes, dropout_prob)
try:
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded successfully. Expected input size: {input_size}")
except Exception as e:
    print(f"Error loading model state dictionary: {e}")
    if "size mismatch" in str(e):
        print("This often means the model architecture (input_size, hidden_size, num_layers, num_classes) defined here")
        print(f"(Expected input_size={input_size}) does not match the architecture of the saved model file.")
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

# --- >>> CHANGE 3: Use the Correct Normalization Function <<< ---
# This function MUST be identical to the one used for creating the 3D dataset
def normalize_landmarks(landmarks, image_shape):
    """
    Normalizes landmarks relative to the hand's bounding box.
    Includes Z coordinate if USE_Z_COORDINATE is True (which it is here).
    Returns a list of features (63) or None if normalization fails.
    """
    if not landmarks:
        return None

    # Extract coordinates into lists
    x_coords_px = [lm.x * image_shape[1] for lm in landmarks.landmark] # Pixel coords
    y_coords_px = [lm.y * image_shape[0] for lm in landmarks.landmark]
    # Z coordinate from MediaPipe is relative depth to wrist, usually needs less normalization
    z_coords = [lm.z for lm in landmarks.landmark]

    if not x_coords_px or not y_coords_px: # Should not happen if landmarks exist, but check
        return None

    # Calculate bounding box
    x_min, x_max = min(x_coords_px), max(x_coords_px)
    y_min, y_max = min(y_coords_px), max(y_coords_px)

    # Avoid division by zero for degenerate bounding boxes (e.g., hand at edge)
    width = x_max - x_min
    height = y_max - y_min
    if width == 0 or height == 0:
        # print(f"Warning: Degenerate bounding box detected (width={width}, height={height}). Skipping normalization for this hand.")
        return None # Indicate failure

    # Normalize x, y relative to the bounding box
    normalized_features = []
    for i, lm in enumerate(landmarks.landmark):
        # Normalize X, Y relative to bounding box
        norm_x = (x_coords_px[i] - x_min) / width
        norm_y = (y_coords_px[i] - y_min) / height
        normalized_features.extend([norm_x, norm_y])
        # Append Z coordinate
        # Z is often used directly or normalized differently (e.g., scaling)
        # Here, we include it as is, matching the likely dataset creation logic.
        normalized_features.append(z_coords[i])

    # Final check on feature count - should be 63 now
    if len(normalized_features) != FEATURES_PER_HAND:
        print(f"Warning in normalize_landmarks: Incorrect feature count after normalization: {len(normalized_features)}, expected {FEATURES_PER_HAND}.")
        return None # Return None on error

    return normalized_features
# --- <<< End of CHANGE 3 >>> ---


# --- Live Test Initialization ---
cap = cv2.VideoCapture(0) # Use 0 for default webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Deques for managing sequences and stability
sequence_buffer = deque(maxlen=SEQUENCE_LENGTH) # Stores feature vectors for recent frames
letter_history = deque(maxlen=stable_threshold) # Stores recent non-J/Z predictions for stability check

# State variables (remain the same)
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

print("\n--- Starting Live Recognition (3D) ---")
print(f"Model: {BEST_MODEL_PATH}")
print(f"Confidence Threshold: {PREDICTION_THRESHOLD}")
print(f"Stability Threshold (frames): {stable_threshold}")
print(f"Cooldown Time (seconds): {cooldown_time}")
# print("Press 'A' to analyze current sentence with Gemini.") # Uncomment if using API
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
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = hands.process(image_rgb)
    image_rgb.flags.writeable = True

    # --- 3. Cooldown Check ---
    if cooldown_active and frame_time - cooldown_start >= cooldown_time:
        cooldown_active = False
        candidate_letter = None

    # --- 4. Landmark Extraction and Feature Processing ---
    # --- >>> CHANGE 4: Update Feature Vector Population <<< ---
    frame_features = np.zeros(TARGET_FEATURES_PER_FRAME, dtype=np.float32) # Initialize 126 zeros
    hands_detected_this_frame = bool(results.multi_hand_landmarks)
    detected_hands_data = [] # Store tuples of (wrist_x, normalized_features) for sorting

    if hands_detected_this_frame:
        no_hand_count = 0 # Reset no-hand counter

        # Process landmarks for up to 2 detected hands
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
            # Draw landmarks and connections on the display frame
            mp_drawing.draw_landmarks(
                display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # --- Use the correct normalize_landmarks function ---
            norm_features = normalize_landmarks(hand_landmarks, (H, W))
            if norm_features:
                # Get wrist x-coordinate for sorting
                wrist_x = hand_landmarks.landmark[0].x
                detected_hands_data.append((wrist_x, norm_features))

                # Draw bounding box (optional, based on 2D coords for simplicity)
                x_coords = [lm.x * W for lm in hand_landmarks.landmark]
                y_coords = [lm.y * H for lm in hand_landmarks.landmark]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                x1, y1 = int(x_min) - 10, int(y_min) - 10
                x2, y2 = int(x_max) + 10, int(y_max) + 10
                box_color = (0, 255, 0) if hand_idx == 0 else (0, 200, 255)
                cv2.rectangle(display_frame, (max(0, x1), max(0, y1)), (min(W, x2), min(H, y2)), box_color, 2)
            # else: # Optional: Handle normalization failure if needed
                # print(f"Normalization failed for hand {hand_idx}")

        # Sort detected hands by horizontal position (leftmost first)
        detected_hands_data.sort(key=lambda item: item[0])

        # Populate the frame_features array (now 126 features)
        num_hands_to_assign = min(len(detected_hands_data), 2)
        for i in range(num_hands_to_assign):
            hand_features = detected_hands_data[i][1] # Get the normalized features (should be 63)
            start_index = i * FEATURES_PER_HAND
            end_index = start_index + FEATURES_PER_HAND
            # Ensure hand_features has the correct length before assignment
            if len(hand_features) == FEATURES_PER_HAND:
                 frame_features[start_index:end_index] = hand_features
            else:
                 # This shouldn't happen if normalize_landmarks works correctly, but good to check
                 print(f"Warning: Hand {i} features had unexpected length {len(hand_features)}, expected {FEATURES_PER_HAND}. Skipping assignment.")

    # --- <<< End of CHANGE 4 >>> ---
    else:
        # No hands detected
        no_hand_count += 1
        letter_history.clear()
        candidate_letter = None

    # Add the processed features (or zeros) to the sequence buffer
    sequence_buffer.append(frame_features)

    # --- 5. Prediction ---
    prediction = None
    current_prediction = None
    current_confidence = 0.0 # Initialize confidence

    if hands_detected_this_frame and len(sequence_buffer) == SEQUENCE_LENGTH:
        try:
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
                        current_prediction = reverse_label_map[pred_idx]
                        current_confidence = conf_val # Store confidence

                        # --- Draw Prediction Text (Near Hand) ---
                        # (Drawing logic remains the same, uses current_prediction and current_confidence)
                        if results.multi_hand_landmarks: # Check again if hands still exist
                            try:
                                first_hand = results.multi_hand_landmarks[0]
                                x_coords = [lm.x for lm in first_hand.landmark]
                                y_coords = [lm.y for lm in first_hand.landmark]
                                x_min, y_min = min(x_coords), min(y_coords)
                                x1_text = max(0, int(x_min * W) - 10)
                                y1_text = max(0, int(y_min * H) - 10)
                                text_to_display = f"{current_prediction} ({current_confidence:.2f})"
                                text_size, _ = cv2.getTextSize(text_to_display, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                                text_x = x1_text
                                text_y = y1_text - 10 if y1_text - 10 > text_size[1] else y1_text + text_size[1] + 30
                                cv2.rectangle(display_frame, (text_x, text_y - text_size[1] - 5),
                                              (text_x + text_size[0], text_y + 5), (0, 255, 0), -1)
                                cv2.putText(display_frame, text_to_display, (text_x, text_y),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                            except IndexError:
                                 pass # Hand might have disappeared between detection and drawing
                        # --- End Drawing Prediction Text ---

                        # --- Handle J/Z or set prediction for stability check ---
                        # (Logic remains the same)
                        if current_prediction in ['J', 'Z']: # Check for Z as well if it's included
                            if not cooldown_active and (not sentence or sentence[-1] != current_prediction):
                                sentence += current_prediction
                                current_letter = current_prediction
                                cooldown_active = True
                                cooldown_start = frame_time
                                print(f"Added (Motion): {current_prediction} | Sentence: {sentence}")
                                letter_history.clear()
                                candidate_letter = None
                            prediction = None # Bypass stability
                        else:
                            prediction = current_prediction # For stability check

        except Exception as e:
            print(f"Prediction error: {e}")
            traceback.print_exc() # Print full traceback for prediction errors
            prediction = None

    # --- 6. Letter Stability Check (for non-J/Z) ---
    # (Logic remains the same)
    if prediction:
        letter_history.append(prediction)
        if len(letter_history) >= stable_threshold:
            try:
                 most_common = max(set(letter_history), key=list(letter_history).count)
                 if list(letter_history).count(most_common) >= int(0.85 * stable_threshold):
                     if candidate_letter != most_common:
                         candidate_letter = most_common
                         letter_hold_start = frame_time
                     else:
                         held_time = frame_time - letter_hold_start
                         if held_time >= required_hold_time and not cooldown_active:
                             if not sentence or sentence[-1] != most_common:
                                 sentence += most_common
                                 current_letter = most_common
                                 cooldown_active = True
                                 cooldown_start = frame_time
                                 print(f"Added: {most_common} | Sentence: {sentence}")
                                 letter_history.clear()
                                 candidate_letter = None
            except ValueError:
                 pass

    # --- 7. Word Break Detection ---
    # (Logic remains the same)
    if no_hand_count >= PAUSE_THRESHOLD_FRAMES and sentence and not sentence.endswith(" "):
        message_text = "Word break!"
        message_until = frame_time + 2.0
        sentence += " "
        print("Added: [SPACE]")
        current_letter = None
        candidate_letter = None
        letter_history.clear()

    # --- 8. Drawing Annotations ---
    # (Drawing logic for FPS, Hand Count, Candidate, Cooldown, Sentence, Messages remains the same)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # FPS
    fps = frame_count / (frame_time - start_time) if (frame_time - start_time) > 0 else 0
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (W - 100, 30), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    # Hand Count
    hands_text = f"Hands: {len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0}"
    cv2.putText(display_frame, hands_text, (10, 30), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    # Candidate
    if candidate_letter:
        held_time = frame_time - letter_hold_start
        hold_status = f"Candidate: {candidate_letter} ({held_time:.1f}s)"
        cv2.putText(display_frame, hold_status, (10, 60), font, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    # Cooldown
    if cooldown_active:
        remaining = max(0, cooldown_time - (frame_time - cooldown_start))
        cooldown_text = f"Cooldown: {remaining:.1f}s"
        cv2.putText(display_frame, cooldown_text, (10, 90), font, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
    # Sentence
    sentence_display = f"Sentence: {sentence}"
    (tw, th), _ = cv2.getTextSize(sentence_display, font, 0.9, 2)
    sx, sy = 10, H - 20
    cv2.rectangle(display_frame, (sx - 5, sy - th - 10), (sx + tw + 5, sy + 5), (50, 50, 50), -1)
    cv2.putText(display_frame, sentence_display, (sx, sy), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    # Message
    if message_text and frame_time < message_until:
        (mw, mh), _ = cv2.getTextSize(message_text, font, 1.0, 2)
        mx = (W - mw) // 2 ; my = 60
        cv2.rectangle(display_frame, (mx - 10, my - mh - 10), (mx + mw + 10, my + 10), (0, 0, 0), -1)
        cv2.putText(display_frame, message_text, (mx, my), font, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    elif frame_time >= message_until:
        message_text = ""

    # API Log Display 
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
    key = cv2.waitKey(1) & 0xFF

    # Analyze sentence with API ('a' key) - Uncomment if using API
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
    if key == ord('q') or key == 27:
        print("\nExiting...")
        break

# --- 11. Cleanup ---
cap.release()
cv2.destroyAllWindows()
for i in range(5): cv2.waitKey(1)
if 'hands' in locals() and hands:
    hands.close()
print("Resources released.")
