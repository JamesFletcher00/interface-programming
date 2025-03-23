import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Modules
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_tasks = mp.tasks

# Load MediaPipe Gesture Recognizer Task
BaseOptions = mp_tasks.BaseOptions
GestureRecognizer = mp_tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp_tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp_tasks.vision.RunningMode


class GestureRecognition:
    def __init__(self, model_path):
        self.options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE  # ✅ Fix: IMAGE mode avoids timestamp issues
        )
        self.recognizer = GestureRecognizer.create_from_options(self.options)

    def recognize_gesture(self, image):
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            result = self.recognizer.recognize(mp_image)
            gestures = []
            if result.gestures:
                for gesture in result.gestures:
                    name = gesture[0].category_name
                    score = gesture[0].score
                    gestures.append((name, score))

            return gestures
        except Exception as e:
            print(f"Error in Gesture Recognition: {e}")
            return []


class FaceRecognition:
    def __init__(self):
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )

    def detect_face(self, image):
        return self.face_detection.process(image)


class HandRecognition:
    def __init__(self):
        self.hands = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect_hands(self, image):
        return self.hands.process(image)


# Function to check if the index finger is pointing up
def is_pointing_up(hand_landmarks):
    if hand_landmarks:
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        return index_finger_tip.y < thumb_tip.y  # Finger tip must be above thumb tip
    return False


# Function to check if finger is in a color box
def check_color_selection(x, y, color_boxes):
    for x1, y1, x2, y2, color in color_boxes:
        if x1 < x < x2 and y1 < y < y2:
            return color  # Return the new selected color
    return None


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    gesture_recognition = GestureRecognition(model_path="gesture_recognizer.task")
    face_recognition = FaceRecognition()
    hand_recognition = HandRecognition()

    # Color selection setup
    selected_color = (255, 0, 0)  # Default color: Blue
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]  # Blue, Green, Red, Yellow, Purple
    color_boxes = [(i * 100, 0, (i + 1) * 100, 50, colors[i]) for i in range(len(colors))]

    # Drawing setup
    drawing = False
    prev_x, prev_y = None, None
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)  # Flip for a mirror effect
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect hands
        hand_results = hand_recognition.detect_hands(rgb_image)

        # Detect gestures
        gestures = gesture_recognition.recognize_gesture(rgb_image)

        # Detect faces
        face_results = face_recognition.detect_face(rgb_image)

        # ✅ Check if Open Palm gesture is detected (to clear screen)
        clear_screen = False
        for gesture, score in gestures:
            if gesture == "Open_Palm" and score >= 0.5:  # ✅ Fix comparison
                clear_screen = True
                break  # Stop checking once Open Palm is detected

        # ✅ Reset the canvas if Open Palm is detected
        if clear_screen:
            canvas[:] = 0  # Reset drawing
            prev_x, prev_y = None, None  # Prevents leftover lines
            print("Canvas cleared!")  # Debugging message

        # Draw color selection boxes
        for x1, y1, x2, y2, color in color_boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)

        # Display recognized gestures on screen
        for i, (gesture, score) in enumerate(gestures):
            cv2.putText(image, f"{gesture} ({score:.2f})", (50, 100 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw face detection results
        if face_results and face_results.detections:
            for detection in face_results.detections:
                mp_drawing.draw_detection(image, detection)

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Get fingertip coordinates
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_finger_tip.x * 640), int(index_finger_tip.y * 480)

                # Check if finger touches a color box
                new_color = check_color_selection(x, y, color_boxes)
                if new_color:
                    selected_color = new_color  # Change drawing color

                # Check if the finger is pointing up to start drawing
                if is_pointing_up(hand_landmarks):
                    drawing = True
                else:
                    drawing = False
                    prev_x, prev_y = None, None  # Reset drawing position

                # Draw only if pointing up
                if drawing:
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), selected_color, 5)
                    prev_x, prev_y = x, y

        # Overlay the drawing on the image
        output = cv2.addWeighted(image, 1, canvas, 0.5, 0)

        # Show the output
        cv2.imshow("Hand Gesture Drawing & Face Recognition", output)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
