import pickle
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import time
from analyze import analyze_asl
import threading

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

with open('models/label_map_alphabet_both.pickle', 'rb') as f:
    label_info = pickle.load(f)
    label_map = label_info['label_map_alphabet']
    reverse_label_map = label_info['reverse_label_map_alphabet']

model = HandGestureCNN(len(label_map))
model.load_state_dict(torch.load('models/best_cnn_model_alphabet_both.pth', map_location=torch.device('cpu')))
model.eval()
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)

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
required_hold_time = 0.8
stable_threshold = 8
# required_hold_time = 1.0
# stable_threshold = 10
letter_history = []
frame_counter = 0
message_text = ""
message_until = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

        input_tensor = torch.FloatTensor(np.array(data_aux).reshape(1, 1, 21, 2))
        with torch.inference_mode():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            predicted_class = predicted.item()
            confidence_value = confidence.item()
            prediction = reverse_label_map[predicted_class]

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
        no_hand_count += 1
        if no_hand_count >= pause_threshold and len(sentence) > 0 and sentence[-1] != " ":
            message_text = "Word break detected!"
            message_until = time.time() + 2.5
            sentence += " "
            current_letter = None
            candidate_letter = None

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (255, 200, 100)
    bg_color = (30, 30, 30)

    sentence_text = f"Sentence: {sentence.strip()}"
    (text_width, text_height), _ = cv2.getTextSize(sentence_text, font, font_scale, font_thickness)
    x, y = 10, H - 30
    cv2.rectangle(frame, (x - 5, y - text_height - 10), (x + text_width + 5, y + 5), bg_color, -1)
    cv2.putText(frame, sentence_text, (x, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

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



    cv2.imshow('Live ASL Recognition', frame)
    key = cv2.waitKey(1) & 0xFF
    
    def analyze_and_log(text):
        analysis = analyze_asl(text)
        print(analysis)
        sentence_log.append(f"{analysis}")
        if len(sentence_log) > 3:
            sentence_log.pop(0)


    if key == ord('a'):
        if sentence.strip():
            threading.Thread(target=analyze_and_log, args=(sentence.strip(),)).start()
            sentence = ""
            current_letter = None
            candidate_letter = None
            letter_history.clear()

    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
