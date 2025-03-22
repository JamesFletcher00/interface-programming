import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Modules
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load MediaPipe Gesture Recognizer Task
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class GestureRecognition:
    def __init__(self, model_path):
        self.options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE
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


# Drawing Canvas
canvas = None
selected_color = (0, 0, 255)  # Default color: Red


def draw_color_boxes(frame):
    """Draws color selection boxes on the screen."""
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]  # Red, Green, Blue, Yellow, Purple
    box_size = 60
    for i, color in enumerate(colors):
        x1, y1 = i * box_size + 10, 10
        x2, y2 = x1 + box_size, y1 + box_size
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    return colors


def main():
    global canvas, selected_color

    # Initialize the camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize recognition modules
    gesture_recognition = GestureRecognition(model_path="gesture_recognizer.task")
    face_recognition = FaceRecognition()
    hand_recognition = HandRecognition()

    canvas = np.zeros((480, 640, 3), dtype=np.uint8)  # Transparent canvas for drawing

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)  # Flip for a mirror effect
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert for MediaPipe

        # Process hand detection
        hand_results = hand_recognition.detect_hands(rgb_image)

        # Draw color selection boxes
        colors = draw_color_boxes(image)

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Get index finger tip (landmark 8)
                index_finger = hand_landmarks.landmark[8]
                x, y = int(index_finger.x * 640), int(index_finger.y * 480)

                # Check if the finger is pointing inside a color box (color selection)
                for i, color in enumerate(colors):
                    x1, y1 = i * 60 + 10, 10
                    x2, y2 = x1 + 60, y1 + 60
                    if x1 < x < x2 and y1 < y < y2:
                        selected_color = color  # Change selected color

                # If finger is pointing upwards, allow drawing
                if y < 400:  # Ensure it's not near bottom of screen
                    cv2.circle(canvas, (x, y), 5, selected_color, -1)  # Draw point

        # Combine canvas and image
        output = cv2.addWeighted(image, 0.7, canvas, 0.3, 0)

        # Display the frame
        cv2.imshow("Hand Gesture Drawing", output)

        # Exit on 'ESC' key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
