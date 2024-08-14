import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('model.keras')

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

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Determine the hand side
            hand_label = handedness.classification[0].label  # 'Left' or 'Right'

            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            x_ = []
            y_ = []

            # Extract and normalize landmarks
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

            # Predict gesture
            data_aux = np.array([data_aux])
            prediction = model.predict(data_aux)

            predicted_character = labels_dict[np.argmax(prediction[0])]
            print(f"Hand: {hand_label}, Prediction: {predicted_character}")

            # Draw bounding box around hand
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, f"{predicted_character} ({hand_label})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                        (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
