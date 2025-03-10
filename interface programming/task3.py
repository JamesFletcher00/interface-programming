import cv2
import mediapipe as mp

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

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert to RGB and process image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Convert back to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        gesture = "No Hand Detected"  # Default state

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = image.shape

                # Get key landmark positions
                fingertips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
                bases = [6, 10, 14, 18]  # Base of each finger

                extended_fingers = 0

                for tip, base in zip(fingertips, bases):
                    tip_y = hand_landmarks.landmark[tip].y * h
                    base_y = hand_landmarks.landmark[base].y * h

                    if tip_y < base_y:  # If fingertip is above base, finger is extended
                        extended_fingers += 1

                # Determine gesture
                if extended_fingers >= 3:  
                    gesture = "Open Palm"
                else:
                    gesture = "Closed Fist"

                # Draw landmarks
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Flip the image for a selfie-view display (before adding text)
        image = cv2.flip(image, 1)

        # Draw text on flipped image (text no longer mirrored)
        cv2.putText(image, gesture, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the processed frame
        cv2.imshow('Hand Gesture Recognition', image)

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
