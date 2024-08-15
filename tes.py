import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

# Load the model
model = tf.keras.models.load_model('model.h5')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Detect up to 2 hands
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

labels_dict = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9'}

accumulated_predictions = ""  # String to accumulate predictions
last_update_time = time.time()  # Record the last time predictions were updated
update_interval = 2  # Interval to update the display in seconds
update_flag = False  # Flag to indicate if 's' has been pressed
initial_display = False  # Flag to indicate if the initial prediction display has been shown
current_prediction = ""  # Track the most recent prediction
flash_duration = 0.1  # Duration of the flash effect in seconds
flash_on = False  # Flag to control the flash effect
hand_detected = True  # To track if a hand was detected in the previous frame
underscore_added = False  # Flag to ensure '_' is added only once

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_detected = True  # Hand is detected
        underscore_added = False  # Reset underscore flag when hand is detected
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label  # 'Left' or 'Right'

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            x_ = []
            y_ = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            x_min, x_max = min(x_), max(x_)
            y_min, y_max = min(y_), max(y_)

            data_aux = []
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                norm_x = (x - x_min) / (x_max - x_min) if x_max > x_min else 0
                norm_y = (y - y_min) / (y_max - y_min) if y_max > y_min else 0
                data_aux.append(norm_x)
                data_aux.append(norm_y)

            data_aux = np.array([data_aux])
            prediction = model.predict(data_aux)

            predicted_character = labels_dict[np.argmax(prediction[0])]
            current_prediction = predicted_character
            print(f"Hand: {hand_label}, Prediction: {predicted_character}")

            x1 = int(min(x_) * W) - 20
            y1 = int(min(y_) * H) - 20
            x2 = int(max(x_) * W) + 20
            y2 = int(max(y_) * H) + 20

            if flash_on:
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1 + 5, y1 + 5), (x2 - 5, y2 - 5), (255, 255, 255), -1)
                alpha = 0.4
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                flash_on = False

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, f"{predicted_character} ({hand_label})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                        (0, 0, 0), 3, cv2.LINE_AA)

    else:
        if hand_detected and not underscore_added:  # Only add '_' once when hand is no longer detected
            accumulated_predictions += "_"
            hand_detected = False  # Set flag to prevent multiple underscores
            underscore_added = True  # Ensure '_' is added only once until hand is detected again

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        update_flag = True
        last_update_time = time.time()
        initial_display = True
        flash_on = True

    if update_flag:
        current_time = time.time()
        if hand_detected and (initial_display or (current_time - last_update_time >= update_interval)):
            if initial_display:
                accumulated_predictions = current_prediction
            else:
                accumulated_predictions += current_prediction

            result_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(result_frame, f"Predictions: {accumulated_predictions}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('Prediction Result', result_frame)

            last_update_time = current_time
            initial_display = False
            flash_on = True

cap.release()
cv2.destroyAllWindows()
