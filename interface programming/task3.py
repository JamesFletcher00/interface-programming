import cv2
import mediapipe as mp
import time
from collections import deque

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Hand landmark mapping
FINGERS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
    "wrist": [0]
}

# Start video capture
cap = cv2.VideoCapture(0)

# Initialize Hands detection
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    prev_x = None  # Store previous wrist x-position
    movement_queue = deque(maxlen=5)  # Stores last few movements
    wave_threshold = 3  # Minimum alternating movements for a wave
    min_movement_distance = 15  # Minimum pixels moved to count as a wave
    last_wave_time = 0  # Timestamp of last detected wave
    wave_cooldown = 1  # Cooldown duration

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Flip image for selfie view
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        gesture = "No Hand Detected"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = image.shape
                wrist_x = int(hand_landmarks.landmark[FINGERS["wrist"][0]].x * w)

                # Detect extended fingers using distance
                extended_fingers = {}
                for finger, points in FINGERS.items():
                    if finger == "wrist":
                        continue
                    tip_y = hand_landmarks.landmark[points[-1]].y * h
                    base_y = hand_landmarks.landmark[points[0]].y * h
                    mid_y = hand_landmarks.landmark[points[1]].y * h
                    extended_fingers[finger] = (tip_y < base_y) and (tip_y < mid_y)

                # Determine hand gesture
                if extended_fingers["index"] and not any(
                        extended_fingers[f] for f in ["thumb", "middle", "ring", "pinky"]):
                    gesture = "Point"
                elif extended_fingers["index"] and extended_fingers["middle"] and not any(
                        extended_fingers[f] for f in ["thumb", "ring", "pinky"]):
                    gesture = "Peace"
                elif all(extended_fingers.values()):
                    gesture = "Open Palm"
                elif not any(extended_fingers.values()):
                    gesture = "Closed Fist"
                elif extended_fingers["index"] and extended_fingers["pinky"] and not any(
                        extended_fingers[f] for f in ["thumb", "middle", "ring"]):
                    gesture = "Rock On"
                elif extended_fingers["thumb"] and not any(
                        extended_fingers[f] for f in ["index", "middle", "ring", "pinky"]):
                    gesture = "Thumbs Up"

                # Wave detection
                if prev_x is not None:
                    movement = wrist_x - prev_x
                    if abs(movement) > min_movement_distance:
                        movement_queue.append(movement)

                prev_x = wrist_x

                if len(movement_queue) >= wave_threshold and time.time() - last_wave_time > wave_cooldown:
                    gesture = "Waving"
                    last_wave_time = time.time()
                    movement_queue.clear()

                # Draw landmarks
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display detected gesture
        cv2.putText(image, gesture, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Hand Gesture Recognition', image)

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Release resources
cap.release()
cv2.destroyAllWindows()