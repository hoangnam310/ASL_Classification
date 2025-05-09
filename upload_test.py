import pickle
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import json
import time
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any

# Define model paths
MODEL_SAVE_DIR = 'models'
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'best_lstm_model_sequences_sorted.pth')
LABEL_MAP_PATH = os.path.join(MODEL_SAVE_DIR, 'label_map_sequences.pickle')


class HandGestureCNN(nn.Module):
    """
    CNN model for hand gesture recognition.
    Architecture: 2 convolutional layers with max pooling, followed by 3 fully connected layers.
    """
    def __init__(self, num_classes: int):
        super(HandGestureCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 2), padding=(1, 0))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1, 0))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))

        # Calculate size after convolutions and pooling
        self.fc_input_size = 128 * 5 * 1

        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        # Initialize weights for better convergence
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights for better training convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
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


class ASLRecognizer:
    """
    Main ASL recognition class that handles video processing and sign detection.
    """
    def __init__(self, model_path: str = BEST_MODEL_PATH, 
                 label_map_path: str = LABEL_MAP_PATH):
        """
        Initialize the ASL recognizer with model and label mapping.
        
        Args:
            model_path: Path to the trained model file
            label_map_path: Path to the label mapping pickle file
        """
        # Configuration
        self.min_confidence = 0.7  # Minimum confidence for prediction
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load label mapping
        with open(label_map_path, 'rb') as f:
            label_info = pickle.load(f)
            self.label_map = label_info['label_map']
            self.reverse_label_map = label_info['reverse_label_map']
        
        # Initialize model
        self.num_classes = len(self.label_map)
        self.model = self._load_model(model_path)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5, 
            max_num_hands=1
        )
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load and prepare the PyTorch model."""
        model = HandGestureCNN(self.num_classes)
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        except FileNotFoundError:
            print(f"Warning: Model file {model_path} not found. Using untrained model.")
        except Exception as e:
            print(f"Error loading model: {e}. Using untrained model.")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _extract_hand_features(self, hand_landmarks) -> Tuple[List[float], Tuple[int, int, int, int]]:
        """
        Extract normalized hand features and bounding box from landmarks.
        
        Returns:
            Tuple containing:
                - List of normalized hand features
                - Tuple of (x1, y1, x2, y2) bounding box coordinates
        """
        # Extract raw coordinates
        x_coords = [landmark.x for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y for landmark in hand_landmarks.landmark]
        
        # Get min/max for normalization
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Create normalized feature vector
        data_aux = []
        for i in range(len(hand_landmarks.landmark)):
            # Normalize to [0,1] range within hand bounding box
            norm_x = (hand_landmarks.landmark[i].x - x_min) / (x_max - x_min) if x_max > x_min else 0
            norm_y = (hand_landmarks.landmark[i].y - y_min) / (y_max - y_min) if y_max > y_min else 0
            data_aux.append(norm_x)
            data_aux.append(norm_y)
        
        return data_aux, (x_min, y_min, x_max, y_max)
    
    def _predict_gesture(self, features: List[float]) -> Tuple[Optional[str], float]:
        """
        Predict the hand gesture from extracted features.
        
        Returns:
            Tuple containing:
                - Predicted letter (None if error)
                - Confidence score (0.0 if error)
        """
        try:
            # Prepare input for CNN
            input_tensor = np.array(features).reshape(1, 1, 21, 2)
            input_tensor = torch.FloatTensor(input_tensor).to(self.device)
            
            # Make prediction
            with torch.inference_mode():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # Get prediction details
                predicted_class = predicted.item()
                confidence_value = confidence.item()
                
                # Validate prediction is in range
                if predicted_class in self.reverse_label_map:
                    prediction = self.reverse_label_map[predicted_class]
                    return prediction, confidence_value
                else:
                    print(f"Warning: Predicted class {predicted_class} not found in label map")
                    return None, 0.0
                    
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0
    
    def _draw_predictions(self, frame: np.ndarray, bbox: Tuple[float, float, float, float], 
                         prediction: str, confidence: float, frame_info: Dict[str, Any]) -> np.ndarray:
        """
        Draw predictions and information on the frame.
        
        Args:
            frame: Current video frame
            bbox: Bounding box coordinates (x_min, y_min, x_max, y_max) as normalized values
            prediction: Predicted letter
            confidence: Confidence score
            frame_info: Dictionary with frame information
            
        Returns:
            Frame with annotations
        """
        H, W = frame.shape[:2]
        x_min, y_min, x_max, y_max = bbox
        
        # Convert normalized coordinates to pixel values
        x1 = max(0, int(x_min * W) - 10)
        y1 = max(0, int(y_min * H) - 10)
        x2 = min(W, int(x_max * W) + 10)
        y2 = min(H, int(y_max * H) + 10)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Prepare text with confidence
        text = f"{prediction} ({confidence:.2f})"
        
        # Get text size for better positioning
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        
        # Position text above the bounding box
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > text_size[1] else y1 + text_size[1] + 10
        
        # Draw text background
        cv2.rectangle(frame, 
                    (text_x, text_y - text_size[1] - 5),
                    (text_x + text_size[0], text_y + 5),
                    (0, 255, 0), -1)
        
        # Draw text
        cv2.putText(frame, text, (text_x, text_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Add frame counter and current letter status with additional timing information
        text_status = f"Frame: {frame_info['frame_number']}/{frame_info['total_frames']}"
        if frame_info.get('current_letter'):
            hold_time = frame_info.get('hold_time', 0)
            time_left = max(0, frame_info.get('required_hold_time', 0) - hold_time)
            
            text_status += f" | Current: {frame_info['current_letter']} "
            text_status += f"(Hold: {hold_time:.1f}s, Needed: {time_left:.1f}s)"
            
        cv2.putText(frame, text_status, (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Add instruction text
        instruction_text = f"Please hold each sign for at least {frame_info.get('required_hold_time', 1.0):.1f} seconds"
        cv2.putText(frame, instruction_text, (10, H - 20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        return frame
    
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a video file to recognize ASL signs.
        
        Args:
            video_path: Path to the input video file
            output_path: Path to save the processed video (optional)
            
        Returns:
            Dictionary with recognition results
        """
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return {
                "processed_video": video_path,
                "output_video": None,
                "status": "error",
                "error_message": "Could not open video file",
                "detected_sentence": "",
                "letter_detections": []
            }
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # ASL-friendly timing parameters (adjusted for natural signing speeds)
        required_hold_time = 1.0  # Seconds to hold a sign before registering it
        transition_time = 0.5     # Minimum time between registered letters
        cooldown_time = 0.8       # Time after registering a letter before allowing the next one
        
        # Calculate frame thresholds
        stable_threshold = max(int(fps * 0.5), 5)        # Min consecutive frames for stable detection
        hold_threshold = max(int(fps * required_hold_time), 5)  # Frames to hold before registering
        pause_threshold = max(int(fps * 1.5), 15)        # No hand detected frames for word break
        
        # Set up video writer if output path is specified
        out = None
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Variables for tracking
        frame_results = []
        letter_history = []
        letter_detections = []
        current_letter = None
        candidate_letter = None
        letter_stable_count = 0
        letter_hold_start = None
        last_letter_time = 0
        no_hand_count = 0
        cooldown_active = False
        cooldown_start = 0
        
        # Process each frame with progress bar
        # pbar = tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}")
        print(f"Processing: {os.path.basename(video_path)}")
        frame_counter = 0
        
        while True:
            # Read frame
            ret, frame = cap.read()
            # print("Frame read:", ret)
            if not ret:
                break
                
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            frame_time = frame_number / fps  # Time in seconds
            
            # Handle cooldown (time after registering a letter)
            if cooldown_active and (frame_time - cooldown_start) >= cooldown_time:
                cooldown_active = False
            
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(frame_rgb)
            
            current_prediction = None
            current_confidence = 0
            
            if results.multi_hand_landmarks:
                # Reset no-hand counter
                no_hand_count = 0
                
                # Get the first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
                
                # Extract features and get bounding box
                features, bbox = self._extract_hand_features(hand_landmarks)
                
                # Get prediction
                prediction, confidence = self._predict_gesture(features)
                
                if prediction and confidence >= self.min_confidence:
                    current_prediction = prediction
                    current_confidence = confidence
                    
                    # Update letter history for stability tracking
                    letter_history.append((prediction, confidence))
                    # Keep history to a reasonable size
                    if len(letter_history) > stable_threshold * 2:
                        letter_history = letter_history[-stable_threshold * 2:]
                    
                    # Calculate hold time if we have a candidate letter
                    hold_time = 0
                    if candidate_letter and candidate_letter == prediction and letter_hold_start is not None:
                        hold_time = frame_time - letter_hold_start
                    
                    # Draw predictions on frame
                    frame_info = {
                        'frame_number': frame_number,
                        'total_frames': total_frames,
                        'current_letter': candidate_letter,
                        'hold_time': hold_time,
                        'required_hold_time': required_hold_time,
                        'stable_count': letter_stable_count
                    }
                    frame = self._draw_predictions(frame, bbox, prediction, confidence, frame_info)
            else:
                # No hand detected
                no_hand_count += 1
                letter_history = []  # Clear history when no hand is detected
                candidate_letter = None
                letter_hold_start = None
            
            # Analyze letter stability
            if len(letter_history) >= stable_threshold and not cooldown_active:
                # Check if the last N predictions are the same
                recent_letters = [item[0] for item in letter_history[-stable_threshold:]]
                most_common = max(set(recent_letters), key=recent_letters.count)
                
                # If we have a consistent prediction (80% agreement)
                if recent_letters.count(most_common) >= stable_threshold * 0.8:
                    # Start tracking a candidate letter
                    if candidate_letter is None or candidate_letter != most_common:
                        candidate_letter = most_common
                        letter_hold_start = frame_time
                        letter_stable_count = 1
                    else:
                        # Still the same candidate letter, update stable count
                        letter_stable_count += 1
                        
                        # Check if we've held this letter long enough and it's not in cooldown
                        hold_time = frame_time - letter_hold_start
                        if (hold_time >= required_hold_time and 
                            (frame_time - last_letter_time) >= transition_time):
                            
                            # If this is a new letter (different from current) or it's the first letter (current is None)
                            if current_letter is None or most_common != current_letter:
                                # Register the letter
                                letter_detections.append({
                                    "letter": most_common,
                                    "confidence": letter_history[-1][1],
                                    "time": frame_time,
                                    "frame": frame_number,
                                    "hold_time": hold_time
                                })
                                
                                # Update tracking
                                current_letter = most_common
                                last_letter_time = frame_time
                                
                                # Start cooldown period
                                cooldown_active = True
                                cooldown_start = frame_time
                                
                                print(f"Detected letter: {most_common} (held for {hold_time:.2f}s)")
                else:
                    # Unstable prediction, reset candidate
                    candidate_letter = None
                    letter_hold_start = None
            
            # Check for word break (pause in signing)
            if no_hand_count >= pause_threshold and current_letter is not None:
                # Add a space to indicate word break
                letter_detections.append({
                    "letter": " ",
                    "confidence": 1.0,
                    "time": frame_time,
                    "frame": frame_number,
                    "is_word_break": True
                })
                current_letter = None
                candidate_letter = None
                letter_hold_start = None
                letter_stable_count = 0
                print("Word break detected")
            
            # Save frame result for detailed analysis
            frame_results.append((frame_number, current_prediction, current_confidence))
            
            # Write the frame to output video if specified
            if out:
                out.write(frame)
                
            # Update progress bar
            # pbar.update(1)
            frame_counter += 1
            if frame_counter % 50 == 0:
                print(f"Processed {frame_counter} frames...")
        
        # Close progress bar
        # pbar.close()
        
        # Release resources
        cap.release()
        if out:
            out.release()
        
        # Process letter detections to create a clean sentence
        filtered_detections = self._filter_letter_detections(letter_detections)
        
        # Create the detected sentence
        detected_sentence = ''.join([d["letter"] for d in filtered_detections])
        space_separated_letters = ' '.join([d["letter"] for d in filtered_detections if d["letter"] != " "])
        
        # Generate summary
        summary = {
            "processed_video": video_path,
            "output_video": output_path,
            "status": "success",
            "detected_sentence": detected_sentence,
            "space_separated_letters": space_separated_letters,
            "letter_detections": [
                {
                    "letter": d["letter"],
                    "confidence": round(float(d["confidence"]), 2),
                    "time_seconds": round(d["time"], 2),
                    "frame": d["frame"],
                    "hold_time": round(d.get("hold_time", 0), 2) if "hold_time" in d else None,
                    "is_word_break": d.get("is_word_break", False)
                }
                for d in filtered_detections
            ]
        }
        
        # Print space-separated letters at the end of processing
        print("\n=====================================")
        print("SPACE SEPARATED LETTERS:")
        print(space_separated_letters)
        print("=====================================\n")
        
        return summary
    
    def _filter_letter_detections(self, letter_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter letter detections to remove repetitions and low confidence detections.
        
        Args:
            letter_detections: List of raw letter detections
            
        Returns:
            Filtered list of letter detections
        """
        filtered_detections = []
        prev_letter = None
        
        for detection in letter_detections:
            letter = detection["letter"]
            
            # Skip if it's the same as previous (avoid repetition)
            if letter == prev_letter and letter != " ":
                continue
                
            # Add to filtered list
            filtered_detections.append(detection)
            prev_letter = letter
        
        return filtered_detections


def process_folder(input_folder: str, output_video_folder: str, output_text_folder: str, 
              model_path: str = BEST_MODEL_PATH,
              label_map_path: str = LABEL_MAP_PATH) -> List[Dict[str, Any]]:
    """
    Process all videos in a folder.
    
    Args:
        input_folder: Path to the folder containing input videos
        output_video_folder: Path to save processed videos
        output_text_folder: Path to save text results
        model_path: Path to the model file
        label_map_path: Path to the label mapping file
        
    Returns:
        List of dictionaries with processing results
    """
    # Create output folders if they don't exist
    os.makedirs(output_video_folder, exist_ok=True)
    os.makedirs(output_text_folder, exist_ok=True)
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist")
        os.makedirs(input_folder, exist_ok=True)
        print(f"Created empty input folder '{input_folder}'")
        return []
    
    # Get all video files in input folder
    video_extensions = ['.mp4', '.avi', '.mkv', '.wmv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(list(Path(input_folder).glob(f'**/*{ext}')))
    
    if not video_files:
        print(f"No video files found in {input_folder}")
        return []
    
    print(f"Found {len(video_files)} video files to process")
    
    # Initialize the recognizer
    try:
        recognizer = ASLRecognizer(model_path=model_path, label_map_path=label_map_path)
    except Exception as e:
        print(f"Error initializing recognizer: {e}")
        return []
    
    all_results = []
    all_space_separated_letters = []
    
    # Process each video
    for i, video_path in enumerate(video_files):
        video_path_str = str(video_path)
        print(f"\n[{i+1}/{len(video_files)}] Processing: {video_path.name}")
        
        # Create output path for the video
        output_path = os.path.join(output_video_folder, f"processed_{os.path.basename(video_path_str)}")
        
        # Process the video
        try:
            start_time = time.time()
            result = recognizer.process_video(video_path_str, output_path)
            processing_time = time.time() - start_time
            
            # Add processing time to results
            result["processing_time_seconds"] = processing_time
            
            # Add result to collection
            all_results.append(result)
            all_space_separated_letters.append(result["space_separated_letters"])  # NEW: Track all space-separated letters
            
            # Print summary
            print(f"Completed in {processing_time:.2f} seconds")
            print(f"Detected sentence: {result['detected_sentence']}")
            print(f"Output video saved to: {output_path}")
        except Exception as e:
            print(f"Error processing video {video_path.name}: {e}")
            all_results.append({
                "processed_video": video_path_str,
                "output_video": output_path,
                "status": "error",
                "error_message": str(e),
                "detected_sentence": "",
                "letter_detections": []
            })
    
    # Save consolidated JSON result
    results_path = os.path.join(output_text_folder, "asl_recognition_results.json")
    try:
        with open(results_path, 'w') as f:
            json.dump({
                "total_videos": len(video_files),
                "processing_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "all_space_separated_letters": all_space_separated_letters,  # NEW: Include all space-separated outputs
                "results": all_results
            }, f, indent=2)
        print(f"\nProcessing complete! Results saved to {results_path}")
        
        # Print combined space-separated letters from all videos
        # if all_space_separated_letters:
        #     print("\n=====================================")
        #     print("ALL VIDEOS - SPACE SEPARATED LETTERS:")
        #     for i, letters in enumerate(all_space_separated_letters):
        #         print(f"Video {i+1}: {letters}")
        #     print("=====================================\n")

    except Exception as e:
        print(f"Error saving results to {results_path}: {e}")
    
    return all_results


def main():
    """Main entry point of the program."""
    try:
        # Define default folder paths
        input_folder = "input_video"
        output_folder = "output_folder"
        
        # Create subdirectories for videos and text results
        output_video_folder = os.path.join(output_folder, "video_output_folder")
        output_text_folder = os.path.join(output_folder, "text_result")
        
        print("ASL Recognition System")
        print("=====================")
        print(f"Input folder: {input_folder}")
        print(f"Output video folder: {output_video_folder}")
        print(f"Output text folder: {output_text_folder}")
        print("=====================")
        
        # Process all videos in the folder
        results = process_folder(
            input_folder, 
            output_video_folder, 
            output_text_folder,
            model_path=BEST_MODEL_PATH,
            label_map_path=LABEL_MAP_PATH
        )
        
        return results
    
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    main()