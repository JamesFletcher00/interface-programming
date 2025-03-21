import cv2
import mediapipe as mp

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
            # Convert the image to MediaPipe Image format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            result = self.recognizer.recognize(mp_image)

            # Display gesture recognition result
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


# Main program loop
def main():
    # Initialize the camera and capture video
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize all recognition modules
    gesture_recognition = GestureRecognition(model_path="gesture_recognizer.task")
    face_recognition = FaceRecognition()
    hand_recognition = HandRecognition()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a selfie-view display
        image = cv2.flip(image, 1)

        # Convert BGR to RGB (required for MediaPipe)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process face and hand detections
        face_results = face_recognition.detect_face(rgb_image)
        hand_results = hand_recognition.detect_hands(rgb_image)

        # Gesture recognition
        gestures = gesture_recognition.recognize_gesture(rgb_image)

        # Draw face detection results
        if face_results.detections:
            for detection in face_results.detections:
                mp_drawing.draw_detection(image, detection)

        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # Display gesture recognition results
        for gesture, score in gestures:
            cv2.putText(image, f"{gesture} ({score:.2f})", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame with all detections
        cv2.imshow("Face, Hand, and Gesture Recognition", image)

        # Exit loop when 'ESC' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
