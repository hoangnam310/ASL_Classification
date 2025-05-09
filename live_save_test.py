import pickle
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import time
from collections import deque
import os
import threading
from analyze_gemini import analyze_asl_gemini
import argparse
from datetime import datetime

MODEL_SAVE_DIR = './models' 
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_lstm_model_sequences_sorted.pth')
LABEL_MAP_PATH = os.path.join(MODEL_SAVE_DIR, 'label_map_sequences.pickle')

SEQUENCE_LENGTH = 10
NUM_LANDMARKS = 21
FEATURES_PER_LANDMARK = 2  # 2 if 2 dimension
FEATURES_PER_HAND = NUM_LANDMARKS * FEATURES_PER_LANDMARK  # 42
TARGET_FEATURES_PER_FRAME = FEATURES_PER_HAND * 2          # 84 
PREDICTION_THRESHOLD = 0.7  

stable_threshold = 8  # Frames needed for non-J/Z letter stability
required_hold_time = 0.0  # Min time a stable non-J/Z letter needs to be held (can be 0)
cooldown_time = 1.5  # Min time before *any* letter (incl. J/Z) can be added again

def record_with_realtime_recognition(output_dir_raw='./raw_videos', output_dir_annotated='./annotated_videos', model=None, label_map=None, reverse_label_map=None):
    """
    Records video from webcam with real-time ASL recognition.
    Returns the path to the saved video files.
    """
    # Create output directories
    os.makedirs(output_dir_raw, exist_ok=True)
    os.makedirs(output_dir_annotated, exist_ok=True)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 0 is usually the default webcam
    
    # Check if webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return None
    
    # Get webcam properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:  # Sometimes webcams report invalid FPS
        fps = 30
    
    # Generate output filenames with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_output_filename = f"asl_raw_{timestamp}.mp4"
    annotated_output_filename = f"asl_annotated_{timestamp}.mp4"
    raw_output_path = os.path.join(output_dir_raw, raw_output_filename)
    annotated_output_path = os.path.join(output_dir_annotated, annotated_output_filename)
    
    # Initialize video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    raw_out = cv2.VideoWriter(raw_output_path, fourcc, fps, (frame_width, frame_height))
    annotated_out = cv2.VideoWriter(annotated_output_path, fourcc, fps, (frame_width, frame_height))
    
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
    
    # --- Initialize recognition state variables ---
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)  # Stores feature vectors for recent frames
    letter_history = deque(maxlen=stable_threshold)  # Stores recent non-J/Z predictions for stability check
    
    # State variables
    sentence = ""
    subtitles = []  # List to store subtitle entries: (start_frame, end_frame, text)
    no_hand_count = 0
    no_hand_start_frame = 0
    current_letter = None
    candidate_letter = None
    letter_hold_start = 0
    cooldown_active = False
    cooldown_start = 0
    current_frame_idx = 0
    sentence_start_frame = 0
    
    # Variables for displaying recording time
    start_time = time.time()
    frames_recorded = 0
    
    print("\n--- ASL Video Recorder with Real-time Recognition ---")
    print("Press 'q' to stop recording and save the video")
    print("Press 'r' to restart recording")
    print("Press 'space' to pause/resume recording")
    print("Recording will begin shortly...")
    
    # Countdown before starting
    # for i in range(3, 0, -1):
    #     ret, frame = cap.read()
    #     if ret:
    #         countdown_text = f"Recording will start in {i}..."
    #         cv2.putText(frame, countdown_text, (50, frame_height - 50), 
    #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #         cv2.imshow('ASL Video Recorder', frame)
    #         cv2.waitKey(1000)  # Wait 1 second between countdown
    
    recording = True
    restart = False
    
    print("Recording started!")
    start_time = time.time()
    
    # Main recording loop
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image from webcam")
            break
        
        # Get the current time
        frame_time = time.time()
        # Calculate elapsed time for recording
        elapsed_time = frame_time - start_time
        # Increment frame index
        current_frame_idx += 1
        
        # Create a copy for annotation but keep the original for raw recording
        display_frame = frame.copy()
        
        # Record to raw video if recording is active
        if recording:
            raw_out.write(frame)
            frames_recorded += 1
        
        # --- ASL Recognition Code ---
        if recording:
            # --- Hand Detection with MediaPipe ---
            # Convert frame to RGB (MediaPipe expects RGB)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Improve performance by marking image as not writeable
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            # Mark image as writeable again for drawing
            image_rgb.flags.writeable = True
            
            # --- Cooldown Check ---
            if cooldown_active and frame_time - cooldown_start >= cooldown_time:
                cooldown_active = False
                candidate_letter = None  # Clear candidate when cooldown ends
            
            # --- Landmark Extraction and Feature Processing ---
            frame_features = np.zeros(TARGET_FEATURES_PER_FRAME, dtype=np.float32)  # Initialize empty features
            hands_detected_this_frame = bool(results.multi_hand_landmarks)
            
            if hands_detected_this_frame:
                if no_hand_count > 0:  # Hands reappeared after absence
                    no_hand_count = 0
                
                processed_hands = []
                
                # Process landmarks for up to 2 detected hands
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                    # Draw landmarks and connections on the display frame
                    mp_drawing.draw_landmarks(
                        display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    
                    # Extract and normalize features for this hand
                    H, W, _ = frame.shape
                    hand_data = process_hand_landmarks(hand_landmarks, (H, W))
                    if hand_data:
                        processed_hands.append(hand_data)
                        # Draw bounding box around the hand
                        x1, y1 = int(hand_data['x_min']) - 10, int(hand_data['y_min']) - 10
                        x2, y2 = int(hand_data['x_max']) + 10, int(hand_data['y_max']) + 10
                        box_color = (0, 255, 0) if hand_idx == 0 else (0, 200, 255)  # Green for 1st, Yellow for 2nd
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
                if no_hand_count == 0:
                    # First frame with no hands, record the start
                    no_hand_start_frame = current_frame_idx
                no_hand_count += 1
                # Reset stability tracking immediately when hands disappear
                letter_history.clear()
                candidate_letter = None
            
            # Add the processed features (or zeros if no hands) to the sequence buffer
            sequence_buffer.append(frame_features)
            
            # --- Prediction ---
            prediction = None  # Prediction variable for the STABILITY check
            current_prediction = None  # Prediction from this frame's inference
            
            # Only predict if sequence buffer is full AND hands were detected AND model is available
            if hands_detected_this_frame and len(sequence_buffer) == SEQUENCE_LENGTH and model is not None:
                try:
                    # Prepare input tensor for the model
                    input_sequence = np.array(list(sequence_buffer), dtype=np.float32)
                    # Add batch dimension (batch_size=1)
                    input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(device)
                    
                    # Perform inference
                    with torch.inference_mode():  # More efficient than torch.no_grad() for inference
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
                                
                                # Draw prediction text on frame
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
                                                  (text_x + text_size[0], text_y + 5), (0, 255, 0), -1)  # Green background
                                    # Draw prediction text
                                    cv2.putText(display_frame, text_to_display, (text_x, text_y),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)  # Black text
                                except IndexError:
                                    print("Warning: Hand landmarks disappeared unexpectedly during drawing.")
                                
                                # --- Handle J/Z immediate addition OR set prediction for stability ---
                                if current_prediction in ['J']:
                                    # Try to add J/Z immediately if cooldown allows and not duplicate
                                    if not cooldown_active and (not sentence or sentence[-1] != current_prediction):
                                        sentence += current_prediction
                                        current_letter = current_prediction  # Update last added
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
                
                except Exception as e:
                    print(f"Prediction error: {e}")
                    prediction = None  # Ensure prediction is None on error
            
            # --- Letter Stability Check (for non-J/Z) ---
            # This block only runs if 'prediction' is not None (i.e., a non-J/Z letter was predicted)
            if prediction:
                letter_history.append(prediction)  # Add the potential stable letter to history
                
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
                                letter_hold_start = frame_time  # Start timer for hold duration
                            else:
                                # Candidate is still the same, check hold time
                                held_time = frame_time - letter_hold_start
                                if held_time >= required_hold_time and not cooldown_active:
                                    # Check if it's different from the last *added* letter
                                    if not sentence or sentence[-1] != most_common:
                                        # Add the stable letter to the sentence!
                                        sentence += most_common
                                        current_letter = most_common  # Update last added
                                        cooldown_active = True
                                        cooldown_start = frame_time
                                        print(f"Added: {most_common} | Sentence: {sentence}")
                                        # Clear history and candidate after successful addition
                                        letter_history.clear()
                                        candidate_letter = None
                    except ValueError:
                        pass
            
            # --- Word Break and Automatic Analysis Detection ---
            # Check if hands have been absent for enough frames
            pause_threshold_frames = int(PAUSE_THRESHOLD_SECONDS * fps)
            if no_hand_count >= pause_threshold_frames:
                # Only add a space and analyze if there's content and we haven't just analyzed
                if sentence and not sentence.endswith(" "):
                    sentence += " "
                    print("Added: [SPACE]")
                
                # If hands have been absent long enough for analysis threshold
                if no_hand_count >= pause_threshold_frames * 2 and sentence.strip():
                    # This is a significant pause - analyze the current sentence
                    trimmed_sentence = sentence.strip()
                    
                    print(f"\nAnalyzing sentence at frame {current_frame_idx}: '{trimmed_sentence}'")
                    
                    # Record subtitle info: start frame, end frame (current frame), and text
                    if no_hand_start_frame > sentence_start_frame:
                        # Add the subtitle entry
                        analysis_result = analyze_sentence(trimmed_sentence)
                        subtitles.append((sentence_start_frame, no_hand_start_frame, analysis_result))
                        print(f"Added subtitle: '{analysis_result}' (frames {sentence_start_frame}-{no_hand_start_frame})")
                        
                        # Reset state for next sentence
                        sentence = ""
                        sentence_start_frame = current_frame_idx
                        no_hand_count = 0  # Reset to avoid multiple triggers
                        current_letter = None
                        candidate_letter = None
                        letter_history.clear()
                        cooldown_active = False
            elif hands_detected_this_frame and not sentence:
                # Hands appeared and we're starting a new sentence
                sentence_start_frame = current_frame_idx
            
            # --- Display Current Status ---
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Hand Count Display (Top Left)
            hands_text = f"Hands: {len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0}"
            cv2.putText(display_frame, hands_text, (200, 30), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)  # Red text
            
            # Candidate Letter Status (Top Left)
            if candidate_letter:
                held_time = frame_time - letter_hold_start
                hold_status = f"Candidate: {candidate_letter} ({held_time:.1f}s)"
                cv2.putText(display_frame, hold_status, (10, 60), font, 0.7, (255, 255, 0), 2, cv2.LINE_AA)  # Cyan text
            
            # Cooldown Status (Top Left)
            if cooldown_active:
                remaining = max(0, cooldown_time - (frame_time - cooldown_start))
                cooldown_text = f"Cooldown: {remaining:.1f}s"
                cv2.putText(display_frame, cooldown_text, (10, 90), font, 0.7, (0, 165, 255), 2, cv2.LINE_AA)  # Orange text
            
            # Current Sentence Display (Bottom)
            sentence_display = f"Sentence: {sentence}"
            (tw, th), _ = cv2.getTextSize(sentence_display, font, 0.9, 2)
            sx, sy = 10, frame_height - 20  # Position near bottom left
            # Background rectangle for sentence
            cv2.rectangle(display_frame, (sx - 5, sy - th - 10), (sx + tw + 5, sy + 5), (50, 50, 50), -1)  # Dark grey bg
            cv2.putText(display_frame, sentence_display, (sx, sy), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)  # White text
            
            # --- Draw any active subtitle ---
            # Find the subtitle that should be active for this frame
            active_subtitle = None
            for start_frame, end_frame, subtitle_text in subtitles:
                if start_frame <= current_frame_idx <= end_frame + int(fps * 3):  # Display for 3 seconds after end
                    active_subtitle = subtitle_text
                    break
            
            # Draw active subtitle if there is one
            if active_subtitle:
                # Format as centered subtitle at bottom of frame
                (sub_w, sub_h), _ = cv2.getTextSize(active_subtitle, font, 1.2, 3)
                subtitle_x = (frame_width - sub_w) // 2
                subtitle_y = frame_height - 50  # Position above the sentence display
                
                # Draw background
                cv2.rectangle(display_frame, 
                             (subtitle_x - 10, subtitle_y - sub_h - 10),
                             (subtitle_x + sub_w + 10, subtitle_y + 10),
                             (0, 0, 0), -1)  # Black background
                
                # Draw subtitle text
                cv2.putText(display_frame, active_subtitle, (subtitle_x, subtitle_y),
                           font, 1.2, (255, 255, 255), 3, cv2.LINE_AA)  # White text
            
            # Write the annotated frame to output video
            annotated_out.write(display_frame)
        
        # Add recording indicator and timer
        time_text = f"REC {int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d}"
        
        if recording:
            # Add red recording circle
            cv2.circle(display_frame, (30, 20), 10, (0, 0, 255), -1)
            # Add recording time
            cv2.putText(display_frame, time_text, (50, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            # Show "PAUSED" when not recording
            cv2.putText(display_frame, "PAUSED", (50, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
        
        # Add instruction text
        cv2.rectangle(display_frame, (frame_width - 250 - 5, 30 - 20), (frame_width - 250 + 230, 30 + 5), (0, 0, 0), -1)  # Black background
        cv2.putText(display_frame, "Press 'q' to save and exit", (frame_width - 250, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(display_frame, (frame_width - 250 - 5, 50 - 20), (frame_width - 250 + 230, 50 + 5), (0, 0, 0), -1)
        cv2.putText(display_frame, "Press 'r' to restart", (frame_width - 250, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(display_frame, (frame_width - 250 - 5, 70 - 20), (frame_width - 250 + 230, 70 + 5), (0, 0, 0), -1)
        cv2.putText(display_frame, "Press 'space' to pause/resume", (frame_width - 250, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Display the frame
        cv2.imshow('ASL Video Recorder with Recognition', display_frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):  # Quit and save
            break
        elif key == ord('r'):  # Restart recording
            restart = True
            break
        elif key == ord(' '):  # Toggle pause/resume with spacebar
            recording = not recording
            if recording:
                print("Recording resumed")
                start_time = time.time() - elapsed_time  # Adjust start time to account for pause
            else:
                print("Recording paused")
    
    # Release resources
    cap.release()
    raw_out.release()
    annotated_out.release()
    cv2.destroyAllWindows()
    
    # Check if we should restart or finish
    if restart:
        print("\nRestarting recording...")
        return record_with_realtime_recognition("./raw_videos", "./annotated_videos", model, label_map, reverse_label_map)  # Recursive call to restart
    
    # Process any final sentence if it exists
    if sentence.strip():
        trimmed_sentence = sentence.strip()
        print(f"\nAnalyzing final sentence: '{trimmed_sentence}'")
        analysis_result = analyze_sentence(trimmed_sentence)
        # Add final subtitle - display until end of video
        subtitles.append((sentence_start_frame, current_frame_idx, analysis_result))
    
    # Calculate actual recorded duration and FPS
    if frames_recorded > 0:
        actual_duration = elapsed_time if recording else elapsed_time - (time.time() - start_time)
        actual_fps = frames_recorded / actual_duration if actual_duration > 0 else 0
        
        print(f"\n--- Recording Saved ---")
        print(f"Raw output file: {raw_output_path}")
        print(f"Annotated output file: {annotated_output_path}")
        print(f"Duration: {int(actual_duration // 60):02d}:{int(actual_duration % 60):02d}")
        print(f"Frames recorded: {frames_recorded}")
        print(f"Actual FPS: {actual_fps:.1f}")
        
        return {
            'raw_video': raw_output_path,
            'annotated_video': annotated_output_path,
            'subtitles': subtitles
        }
    else:
        print("\nNo frames were recorded.")
        if os.path.exists(raw_output_path):
            os.remove(raw_output_path)
        if os.path.exists(annotated_output_path):
            os.remove(annotated_output_path)
        return None

# def record_webcam_video(output_dir_raw='./raw_videos', output_dir_annotated='./annotated_videos'):
#     """
#     Records video from webcam until 'q' key is pressed.
#     Returns the path to the saved video file.
#     """
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir_raw, exist_ok=True)
#     os.makedirs(output_dir_annotated, exist_ok=True)
    
#     # Initialize webcam
#     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 0 is usually the default webcam
    
#     # Check if webcam is opened successfully
#     if not cap.isOpened():
#         print("Error: Could not open webcam")
#         return None
    
#     # Get webcam properties
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     if fps <= 0:  # Sometimes webcams report invalid FPS
#         fps = 30
    
#     # Generate output filename with timestamp
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_filename = f"asl_recording_{timestamp}.mp4"
#     output_path = os.path.join(output_dir_annotated, output_filename)
    
#     # Initialize video writer
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
#     out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
#     # Variables for displaying recording time
#     start_time = time.time()
#     frames_recorded = 0
    
#     print("\n--- ASL Video Recorder ---")
#     print("Press 'q' to stop recording and save the video")
#     print("Press 'r' to restart recording")
#     print("Press 'space' to pause/resume recording")
#     print("Recording will begin shortly...")
    
#     # Countdown before starting
#     for i in range(3, 0, -1):
#         ret, frame = cap.read()
#         if ret:
#             countdown_text = f"Recording will start in {i}..."
#             cv2.putText(frame, countdown_text, (50, frame_height - 50), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#             cv2.imshow('ASL Video Recorder', frame)
#             cv2.waitKey(1000)  # Wait 1 second between countdown
    
#     recording = True
#     restart = False
    
#     print("Recording started!")
#     start_time = time.time()
    
#     # Main recording loop
#     while True:
#         ret, frame = cap.read()
        
#         if not ret:
#             print("Error: Failed to capture image from webcam")
#             break
        
#         # Add recording indicator and timer
#         elapsed_time = time.time() - start_time
#         time_text = f"REC {int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d}"
        
#         if recording:
#             # Add red recording circle
#             cv2.circle(frame, (30, 20), 10, (0, 0, 255), -1)
#             # Add recording time
#             cv2.putText(frame, time_text, (50, 30), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#             # Write frame to video file
#             out.write(frame)
#             frames_recorded += 1
#         else:
#             # Show "PAUSED" when not recording
#             cv2.putText(frame, "PAUSED", (50, 30), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
#         # Add instruction text
#         # cv2.putText(frame, "Press 'q' to save and exit", (frame_width - 250, 30), 
#         #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#         # cv2.putText(frame, "Press 'r' to restart", (frame_width - 250, 50), 
#         #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#         # cv2.putText(frame, "Press 'space' to pause/resume", (frame_width - 250, 70), 
#         #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
#         # Instruction 1
#         text1 = "Press 'q' to save and exit"
#         (x1, y1) = (frame_width - 250, 30)
#         cv2.rectangle(frame, (x1 - 5, y1 - 20), (x1 + 230, y1 + 5), (0, 0, 0), -1)  # Black background
#         cv2.putText(frame, text1, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#         # Instruction 2
#         text2 = "Press 'r' to restart"
#         (x2, y2) = (frame_width - 250, 50)
#         cv2.rectangle(frame, (x2 - 5, y2 - 20), (x2 + 200, y2 + 5), (0, 0, 0), -1)
#         cv2.putText(frame, text2, (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#         # Instruction 3
#         text3 = "Press 'space' to pause/resume"
#         (x3, y3) = (frame_width - 250, 70)
#         cv2.rectangle(frame, (x3 - 5, y3 - 20), (x3 + 280, y3 + 5), (0, 0, 0), -1)
#         cv2.putText(frame, text3, (x3, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        
#         # Display the frame
#         cv2.imshow('ASL Video Recorder', frame)
        
#         # Check for key presses
#         key = cv2.waitKey(1) & 0xFF
        
#         if key == ord('q'):  # Quit and save
#             break
#         elif key == ord('r'):  # Restart recording
#             restart = True
#             break
#         elif key == ord(' '):  # Toggle pause/resume with spacebar
#             recording = not recording
#             if recording:
#                 print("Recording resumed")
#                 start_time = time.time() - elapsed_time  # Adjust start time to account for pause
#             else:
#                 print("Recording paused")
    
#     # Release resources
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
    
#     # Check if we should restart or finish
#     if restart:
#         print("\nRestarting recording...")
#         return record_webcam_video("./raw_videos", "./annotated_videos")  # Recursive call to restart
    
#     # Calculate actual recorded duration and FPS
#     if frames_recorded > 0:
#         actual_duration = elapsed_time if recording else elapsed_time - (time.time() - start_time)
#         actual_fps = frames_recorded / actual_duration if actual_duration > 0 else 0
        
#         print(f"\n--- Recording Saved ---")
#         print(f"Output file: {output_path}")
#         print(f"Duration: {int(actual_duration // 60):02d}:{int(actual_duration % 60):02d}")
#         print(f"Frames recorded: {frames_recorded}")
#         print(f"Actual FPS: {actual_fps:.1f}")
        
#         return output_path
#     else:
#         print("\nNo frames were recorded.")
#         if os.path.exists(output_path):
#             os.remove(output_path)  # Remove empty file
#         return None

def record_webcam_video(input_path, output_dir='./annotated_videos'):
    """
    Annotates an existing raw video file with overlays like timestamp, instructions, etc.
    Saves the result to the annotated_videos folder.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return None

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open input video")
        return None

    # Get properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Generate output path
    basename = os.path.basename(input_path)
    name, _ = os.path.splitext(basename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{name}_annotated_{timestamp}.mp4"
    output_path = os.path.join(output_dir, output_filename)

    # VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    start_time = time.time()

    print(f"Annotating: {input_path}")
    print(f"Saving to: {output_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed_time = time.time() - start_time
        time_text = f"REC {int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d}"
        cv2.circle(frame, (30, 20), 10, (0, 0, 255), -1)
        cv2.putText(frame, time_text, (50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Instructions
        text1 = "Press 'q' to save and exit"
        cv2.rectangle(frame, (frame_width - 250 - 5, 30 - 20), (frame_width - 250 + 230, 30 + 5), (0, 0, 0), -1)
        cv2.putText(frame, text1, (frame_width - 250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Write to output
        out.write(frame)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Progress: {frame_idx}/{total_frames} frames")

    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Annotation complete.")
    return output_path

# Command line argument parsing
# Replace the existing argument parser section:
parser = argparse.ArgumentParser(description='Process ASL videos and create subtitled output')
parser.add_argument('--input', type=str, help='Path to input video file')
parser.add_argument('--output_dir', type=str, default='./annotated_videos', help='Directory to save output video')
parser.add_argument('--pause_threshold', type=float, default=1.0, 
                   help='Time in seconds without hands to trigger sentence analysis')
parser.add_argument('--record', action='store_true', help='Record a new video from webcam instead of using input file')
parser.add_argument('--recordings_dir', type=str, default='./raw_videos', help='Directory to save recorded videos')
parser.add_argument('--realtime', action='store_true', help='Apply ASL recognition in real-time during recording')
args = parser.parse_args()

# Create output directories if they don't exist
os.makedirs(args.output_dir, exist_ok=True)
if args.record:
    os.makedirs(args.recordings_dir, exist_ok=True)

# Model paths and constants
# MODEL_SAVE_DIR = './models' 
# BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_lstm_model_sequences_sorted.pth')
# LABEL_MAP_PATH = os.path.join(MODEL_SAVE_DIR, 'label_map_sequences.pickle')

# SEQUENCE_LENGTH = 10
# NUM_LANDMARKS = 21
# FEATURES_PER_LANDMARK = 2  # 2 if 2 dimension
# FEATURES_PER_HAND = NUM_LANDMARKS * FEATURES_PER_LANDMARK  # 42
# TARGET_FEATURES_PER_FRAME = FEATURES_PER_HAND * 2          # 84 
# PREDICTION_THRESHOLD = 0.7  
PAUSE_THRESHOLD_SECONDS = args.pause_threshold  # Configurable pause threshold in seconds

# Recognition parameters
# stable_threshold = 8  # Frames needed for non-J/Z letter stability
# required_hold_time = 0.0  # Min time a stable non-J/Z letter needs to be held (can be 0)
# cooldown_time = 1.5  # Min time before *any* letter (incl. J/Z) can be added again

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

input_size = TARGET_FEATURES_PER_FRAME  # 84
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
        normalized_features = normalized_features[:FEATURES_PER_HAND]  # Truncate
        while len(normalized_features) < FEATURES_PER_HAND:  # Pad
            normalized_features.append(0.0)

    return {
        'features': normalized_features,
        'x_min': x_min, 'y_min': y_min,
        'x_max': x_max, 'y_max': y_max
    }

# --- Analyze sentence and return result ---
def analyze_sentence(text_to_analyze):
    try:
        analysis = analyze_asl_gemini(text_to_analyze)
        print(f"API Response: {analysis}")
        return analysis
    except Exception as api_e:
        print(f"Error during API call: {api_e}")
        return "API Error"

# --- Process video and create subtitled output ---
def process_video(input_video_path, output_dir):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate pause threshold in frames
    pause_threshold_frames = int(PAUSE_THRESHOLD_SECONDS * fps)
    
    # Create output video filename
    video_basename = os.path.splitext(os.path.basename(input_video_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_path = os.path.join(output_dir, f"{video_basename}_subtitled_{timestamp}.mp4")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Deques for managing sequences and stability
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)  # Stores feature vectors for recent frames
    letter_history = deque(maxlen=stable_threshold)  # Stores recent non-J/Z predictions for stability check
    
    # State variables
    sentence = ""
    subtitles = []  # List to store subtitle entries: (start_frame, end_frame, text)
    no_hand_count = 0
    no_hand_start_frame = 0
    current_letter = None
    candidate_letter = None
    letter_hold_start = 0
    cooldown_active = False
    cooldown_start = 0
    current_frame_idx = 0
    sentence_start_frame = 0
    
    print(f"\n--- Processing Video: {input_video_path} ---")
    print(f"FPS: {fps}, Resolution: {frame_width}x{frame_height}")
    print(f"Total frames: {total_frames}")
    print(f"Pause threshold: {PAUSE_THRESHOLD_SECONDS}s ({pause_threshold_frames} frames)")
    print("------------------------------------")
    
    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_time = time.time()  # Timestamp for processing
        current_frame_idx += 1
        
        if current_frame_idx % 30 == 0:  # Print progress every 30 frames
            progress = (current_frame_idx / total_frames) * 100
            print(f"Processing: {progress:.1f}% (Frame {current_frame_idx}/{total_frames})")
        
        H, W, _ = frame.shape
        display_frame = frame.copy()  # Create a copy for drawing annotations
        
        # --- MediaPipe Hand Tracking ---
        # Convert frame to RGB (MediaPipe expects RGB)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Improve performance by marking image as not writeable before processing
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)
        # Mark image as writeable again for drawing
        image_rgb.flags.writeable = True
        
        # --- Cooldown Check ---
        if cooldown_active and frame_time - cooldown_start >= cooldown_time:
            cooldown_active = False
            candidate_letter = None  # Clear candidate when cooldown ends
        
        # --- Landmark Extraction and Feature Processing ---
        frame_features = np.zeros(TARGET_FEATURES_PER_FRAME, dtype=np.float32)  # Initialize empty features
        hands_detected_this_frame = bool(results.multi_hand_landmarks)
        
        if hands_detected_this_frame:
            if no_hand_count > 0:  # Hands reappeared after absence
                no_hand_count = 0
            
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
                    box_color = (0, 255, 0) if hand_idx == 0 else (0, 200, 255)  # Green for 1st, Yellow for 2nd
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
            if no_hand_count == 0:
                # First frame with no hands, record the start
                no_hand_start_frame = current_frame_idx
            no_hand_count += 1
            # Reset stability tracking immediately when hands disappear
            letter_history.clear()
            candidate_letter = None
        
        # Add the processed features (or zeros if no hands) to the sequence buffer
        sequence_buffer.append(frame_features)
        
        # --- Prediction ---
        prediction = None  # Prediction variable for the STABILITY check
        current_prediction = None  # Prediction from this frame's inference
        
        # Only predict if sequence buffer is full AND hands were detected
        if hands_detected_this_frame and len(sequence_buffer) == SEQUENCE_LENGTH:
            try:
                # Prepare input tensor for the model
                input_sequence = np.array(list(sequence_buffer), dtype=np.float32)
                # Add batch dimension (batch_size=1)
                input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(device)
                
                # Perform inference
                with torch.inference_mode():  # More efficient than torch.no_grad() for inference
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
                            
                            # Draw prediction text on frame
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
                                              (text_x + text_size[0], text_y + 5), (0, 255, 0), -1)  # Green background
                                # Draw prediction text
                                cv2.putText(display_frame, text_to_display, (text_x, text_y),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)  # Black text
                            except IndexError:
                                print("Warning: Hand landmarks disappeared unexpectedly during drawing.")
                            
                            # --- Handle J/Z immediate addition OR set prediction for stability ---
                            if current_prediction in ['J']:
                                # Try to add J/Z immediately if cooldown allows and not duplicate
                                if not cooldown_active and (not sentence or sentence[-1] != current_prediction):
                                    sentence += current_prediction
                                    current_letter = current_prediction  # Update last added
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
            
            except Exception as e:
                print(f"Prediction error: {e}")
                prediction = None  # Ensure prediction is None on error
        
        # --- Letter Stability Check (for non-J/Z) ---
        # This block only runs if 'prediction' is not None (i.e., a non-J/Z letter was predicted)
        if prediction:
            letter_history.append(prediction)  # Add the potential stable letter to history
            
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
                            letter_hold_start = frame_time  # Start timer for hold duration
                        else:
                            # Candidate is still the same, check hold time
                            held_time = frame_time - letter_hold_start
                            if held_time >= required_hold_time and not cooldown_active:
                                # Check if it's different from the last *added* letter
                                if not sentence or sentence[-1] != most_common:
                                    # Add the stable letter to the sentence!
                                    sentence += most_common
                                    current_letter = most_common  # Update last added
                                    cooldown_active = True
                                    cooldown_start = frame_time
                                    print(f"Added: {most_common} | Sentence: {sentence}")
                                    # Clear history and candidate after successful addition
                                    letter_history.clear()
                                    candidate_letter = None
                except ValueError:
                    pass
        
        # --- Word Break and Automatic Analysis Detection ---
        # Check if hands have been absent for enough frames
        if no_hand_count >= pause_threshold_frames:
            # Only add a space and analyze if there's content and we haven't just analyzed
            if sentence and not sentence.endswith(" "):
                sentence += " "
                print("Added: [SPACE]")
            
            # If hands have been absent long enough for analysis threshold
            if no_hand_count >= pause_threshold_frames * 2 and sentence.strip():
                # This is a significant pause - analyze the current sentence
                trimmed_sentence = sentence.strip()
                
                print(f"\nAnalyzing sentence at frame {current_frame_idx}: '{trimmed_sentence}'")
                
                # Record subtitle info: start frame, end frame (current frame), and text
                if no_hand_start_frame > sentence_start_frame:
                    # Add the subtitle entry
                    analysis_result = analyze_sentence(trimmed_sentence)
                    subtitles.append((sentence_start_frame, no_hand_start_frame, analysis_result))
                    print(f"Added subtitle: '{analysis_result}' (frames {sentence_start_frame}-{no_hand_start_frame})")
                    
                    # Reset state for next sentence
                    sentence = ""
                    sentence_start_frame = current_frame_idx
                    no_hand_count = 0  # Reset to avoid multiple triggers
                    current_letter = None
                    candidate_letter = None
                    letter_history.clear()
                    cooldown_active = False
        elif hands_detected_this_frame and not sentence:
            # Hands appeared and we're starting a new sentence
            sentence_start_frame = current_frame_idx
        
        # --- Display Current Status ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Hand Count Display (Top Left)
        hands_text = f"Hands: {len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0}"
        cv2.putText(display_frame, hands_text, (200, 30), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)  # Red text
        
        # Candidate Letter Status (Top Left)
        if candidate_letter:
            held_time = frame_time - letter_hold_start
            hold_status = f"Candidate: {candidate_letter} ({held_time:.1f}s)"
            cv2.putText(display_frame, hold_status, (10, 60), font, 0.7, (255, 255, 0), 2, cv2.LINE_AA)  # Cyan text
        
        # Cooldown Status (Top Left)
        if cooldown_active:
            remaining = max(0, cooldown_time - (frame_time - cooldown_start))
            cooldown_text = f"Cooldown: {remaining:.1f}s"
            cv2.putText(display_frame, cooldown_text, (10, 90), font, 0.7, (0, 165, 255), 2, cv2.LINE_AA)  # Orange text
        
        # Current Sentence Display (Bottom)
        sentence_display = f"Sentence: {sentence}"
        (tw, th), _ = cv2.getTextSize(sentence_display, font, 0.9, 2)
        sx, sy = 10, H - 20  # Position near bottom left
        # Background rectangle for sentence
        cv2.rectangle(display_frame, (sx - 5, sy - th - 10), (sx + tw + 5, sy + 5), (50, 50, 50), -1)  # Dark grey bg
        cv2.putText(display_frame, sentence_display, (sx, sy), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)  # White text
        
        # --- Draw any active subtitle ---
        # Find the subtitle that should be active for this frame
        active_subtitle = None
        for start_frame, end_frame, subtitle_text in subtitles:
            if start_frame <= current_frame_idx <= end_frame + int(fps * 3):  # Display for 3 seconds after end
                active_subtitle = subtitle_text
                break
        
        # Draw active subtitle if there is one
        if active_subtitle:
            # Format as centered subtitle at bottom of frame
            (sub_w, sub_h), _ = cv2.getTextSize(active_subtitle, font, 1.2, 3)
            subtitle_x = (W - sub_w) // 2
            subtitle_y = H - 50  # Position above the sentence display
            
            # Draw background
            cv2.rectangle(display_frame, 
                         (subtitle_x - 10, subtitle_y - sub_h - 10),
                         (subtitle_x + sub_w + 10, subtitle_y + 10),
                         (0, 0, 0), -1)  # Black background
            
            # Draw subtitle text
            cv2.putText(display_frame, active_subtitle, (subtitle_x, subtitle_y),
                       font, 1.2, (255, 255, 255), 3, cv2.LINE_AA)  # White text
        
        # Write the frame to output video
        out.write(display_frame)
    
    # Process any final sentence if it exists
    if sentence.strip():
        trimmed_sentence = sentence.strip()
        print(f"\nAnalyzing final sentence: '{trimmed_sentence}'")
        analysis_result = analyze_sentence(trimmed_sentence)
        # Add final subtitle - display until end of video
        subtitles.append((sentence_start_frame, current_frame_idx, analysis_result))
    
    # Clean up resources
    cap.release()
    out.release()
    
    print(f"\n--- Video Processing Complete ---")
    print(f"Output saved to: {output_video_path}")
    print(f"Total subtitles generated: {len(subtitles)}")
    
    # Return the path to the output video
    return output_video_path

if __name__ == "__main__":
    try:
      # Add new command line argument for real-time recognition
      # parser = argparse.ArgumentParser(description='Process ASL videos and create subtitled output')
      # parser.add_argument('--input', type=str, help='Path to input video file')
      # parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save output video')
      # parser.add_argument('--pause_threshold', type=float, default=1.0, 
      #                     help='Time in seconds without hands to trigger sentence analysis')
      # parser.add_argument('--record', action='store_true', help='Record a new video from webcam instead of using input file')
      # parser.add_argument('--realtime', action='store_true', help='Apply ASL recognition in real-time during recording')
      # parser.add_argument('--recordings_dir', type=str, default='./recordings', help='Directory to save recorded videos')
      # args = parser.parse_args()

      # Create output directories if they don't exist
      os.makedirs(args.output_dir, exist_ok=True)
      if args.record or args.realtime:
          os.makedirs(args.recordings_dir, exist_ok=True)
      
      input_video_path = None
      
      # Check if we should record a new video with real-time recognition
      if args.realtime:
          print("Starting webcam recording with real-time ASL recognition...")
          result = record_with_realtime_recognition(
              output_dir_raw=args.recordings_dir,
              output_dir_annotated=args.output_dir,
              model=model,
              label_map=label_map,
              reverse_label_map=reverse_label_map
          )
          if not result:
              print("Recording with real-time recognition failed or was canceled.")
              exit(1)
          print(f"Successfully recorded and processed video.")
          print(f"Raw video saved to: {result['raw_video']}")
          print(f"Annotated video with recognition saved to: {result['annotated_video']}")
          # We're done here
          exit(0)
      
      # Check if we should record a new video (without real-time recognition)
      elif args.record:
          print("Starting webcam recording mode...")
        #   input_video_path = record_webcam_video(args.recordings_dir, args.output_dir)
        #   if not input_video_path:
        #       print("Recording failed or was canceled.")
        #       exit(1)
        #   print(f"Successfully recorded video: {input_video_path}")
          result = record_with_realtime_recognition(
            output_dir_raw=args.recordings_dir,
            output_dir_annotated=args.output_dir,
            model=model,
            label_map=label_map,
            reverse_label_map=reverse_label_map
          )
          if not result:
            print("Recording failed or was canceled.")
            exit(1)
          print(f"Successfully recorded video.")
          print(f"Raw video saved to: {result['raw_video']}")
          print(f"Annotated video saved to: {result['annotated_video']}")
          exit(0)
      else:
          # Use provided input file
          if not args.input:
              print("Error: When not recording, --input video file is required")
              parser.print_help()
              exit(1)
          input_video_path = args.input
      
      # Verify input file exists (for non-realtime modes)
      if not os.path.exists(input_video_path):
          print(f"Error: Input video file not found: {input_video_path}")
          exit(1)
      
      # Process the video (for non-realtime modes)
      output_path = process_video(input_video_path, args.output_dir)
      
      # Display final message
      if output_path:
          print(f"\nVideo processing completed successfully.")
          print(f"Output video saved to: {output_path}")
      else:
          print("\nVideo processing failed.")
    except Exception as e:
      import traceback
      print("An unhandled exception occurred:")
      traceback.print_exc()
