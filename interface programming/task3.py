import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Hand recognition class
class HandRecognition:
    def __init__(self):
        self.hands = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect_hands(self, image):
        return self.hands.process(image)


# Function to check if finger is pointing up
def is_pointing_up(hand_landmarks):
    if hand_landmarks:
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        
        # Index finger tip should be higher than the thumb
        return index_finger_tip.y < thumb_tip.y
    return False


# Function to check if finger is in a color box
def check_color_selection(x, y, color_boxes):
    for i, (x1, y1, x2, y2, color) in enumerate(color_boxes):
        if x1 < x < x2 and y1 < y < y2:
            return color  # Return the new selected color
    return None


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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
        hand_results = hand_recognition.detect_hands(rgb_image)

        # Draw color selection boxes
        for x1, y1, x2, y2, color in color_boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)

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
        image = cv2.addWeighted(image, 1, canvas, 0.5, 0)

        # Show the output
        cv2.imshow("Hand Gesture Drawing", image)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
