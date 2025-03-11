import cv2
import mediapipe as mp
import time
from collections import deque

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Start video capture
cap = cv2.VideoCapture(0)

# Initialize Hands detection
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    prev_x = None  # Store previous wrist x-position
    movement_queue = deque(maxlen=5)  # Stores last few movements
    wave_threshold = 2  # Minimum alternating movements for a wave (LOWERED)
    min_movement_distance = 18  # Minimum pixels moved to count as a wave (LOWERED)
    last_wave_time = 0  # Timestamp of last detected wave
    wave_cooldown = 0  # Shorter cooldown (1 sec)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip image for selfie view
        image = cv2.flip(image, 1)

        # Create a separate unflipped version for text
        text_overlay = image.copy()

        # Convert to RGB and process image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Convert back to BGR for OpenCV
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        gesture = "Unknown"  # Default state

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = image.shape

                # Get wrist position (landmark 0) for wave detection
                wrist_x = int(hand_landmarks.landmark[0].x * w)

                # Get key landmark positions for open palm vs. closed fist
                index_tip, middle_tip, ring_tip, pinky_tip = 8, 12, 16, 20
                index_base, middle_base, ring_base, pinky_base = 6, 10, 14, 18

                # Get y-positions of fingertips and their bases
                index_y = hand_landmarks.landmark[index_tip].y * h
                middle_y = hand_landmarks.landmark[middle_tip].y * h
                ring_y = hand_landmarks.landmark[ring_tip].y * h
                pinky_y = hand_landmarks.landmark[pinky_tip].y * h

                index_base_y = hand_landmarks.landmark[index_base].y * h
                middle_base_y = hand_landmarks.landmark[middle_base].y * h
                ring_base_y = hand_landmarks.landmark[ring_base].y * h
                pinky_base_y = hand_landmarks.landmark[pinky_base].y * h

                # Determine if fingers are extended
                index_extended = index_y < index_base_y
                middle_extended = middle_y < middle_base_y
                ring_extended = ring_y < ring_base_y
                pinky_extended = pinky_y < pinky_base_y

                # Determine hand gesture
                if index_extended and middle_extended and not ring_extended and not pinky_extended:
                    gesture = "Peace ✌️"
                elif index_extended and middle_extended and ring_extended and pinky_extended:
                    gesture = "Open Palm"
                elif not index_extended and not middle_extended and not ring_extended and not pinky_extended:
                    gesture = "Closed Fist"

                # Wave detection: track wrist movement
                if prev_x is not None:
                    movement = wrist_x - prev_x  # Change in X position
                    if abs(movement) > min_movement_distance:  # Ignore small movements
                        movement_queue.append(movement)

                prev_x = wrist_x  # Update wrist position

                # Check if alternating left-right movements occurred
                if len(movement_queue) >= wave_threshold:
                    if time.time() - last_wave_time > wave_cooldown:  # Enforce cooldown
                        gesture = "Waving"
                        last_wave_time = time.time()  # Reset cooldown timer
                        movement_queue.clear()  # Reset movement tracking

                # Draw landmarks
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Draw text on the unflipped image
        cv2.putText(text_overlay, gesture, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Merge the text overlay with the flipped image
        image = text_overlay

        # Show the processed frame
        cv2.imshow('Hand Gesture Recognition', image)

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
