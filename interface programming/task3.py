import cv2
import mediapipe as mp
import numpy as np
import pygame  # ✅ For playing audio
import time

#Initialize MediaPipe Modules
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh  # ✅ Added Face Mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_tasks = mp.tasks

#Load MediaPipe Gesture Recognizer Task
BaseOptions = mp_tasks.BaseOptions
GestureRecognizer = mp_tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp_tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp_tasks.vision.RunningMode

#initialize pygame to play audio 
pygame.mixer.init()
mouth_sound = "interface_audio.mp3"  # Change this to your audio file
pygame.mixer.music.load(mouth_sound)

#class for recognising gestures
class GestureRecognition:
    def __init__(self, model_path):
        self.options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE  
        )
        self.recognizer = GestureRecognizer.create_from_options(self.options)

    def recognize_gesture(self, image):
        #recognises the gesture in the image
        #returns a list of detected gestures / confident score
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

#class for detecting landmarks in the face
class FaceRecognition:
    def __init__(self):
        self.face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5)

    def detect_face(self, image):
        return self.face_detection.process(image)

    def detect_face_mesh(self, image):
        return self.face_mesh.process(image)

#class for detecting landmarks in the hand
class HandRecognition:
    def __init__(self):
        self.hands = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect_hands(self, image):
        return self.hands.process(image)


#function to check if the mouth is open
def is_mouth_open(face_landmarks, threshold=0.05):
    if face_landmarks:
        # Get landmarks for upper & lower lips
        upper_lip = face_landmarks.landmark[13]  # Upper lip (middle)
        lower_lip = face_landmarks.landmark[14]  # Lower lip (middle)

      
        mouth_open_ratio = abs(upper_lip.y - lower_lip.y)

        return mouth_open_ratio > threshold  #returns True if mouth is open
    return False


#function to check if the index finger is pointing up
def is_pointing_up(hand_landmarks):
    if hand_landmarks:
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        return index_finger_tip.y < thumb_tip.y  #tip of index finger must be above thumb tip
    return False

#function to check if the thumb is pointing down
#checks if tip of thumb is below base of thumb, finger tip and wrist
def is_thumb_down(hand_landmarks):
    if not hand_landmarks:
        return False

    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    #thumb tip must be below the index finger tip
    thumb_below_index = thumb_tip.y > index_finger_tip.y

    #thumb tip must be below the thumb MCP
    thumb_pointing_down = thumb_tip.y > thumb_mcp.y

    #thumb tip must be below the wrist
    thumb_below_wrist = thumb_tip.y > wrist.y

    return thumb_below_index and thumb_pointing_down and thumb_below_wrist



#function to check if finger is in a color box
def check_color_selection(x, y, color_boxes):
    for x1, y1, x2, y2, color in color_boxes:
        if x1 < x < x2 and y1 < y < y2:
            return color  # return the new selected color
    return None

#main function to capture video and process all logic
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    gesture_recognition = GestureRecognition(model_path="gesture_recognizer.task")
    face_recognition = FaceRecognition()
    hand_recognition = HandRecognition()

    #colour selection setup
    selected_color = (255, 0, 0)  # sets colour by default to blue
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]
    color_boxes = [(i * 100, 0, (i + 1) * 100, 50, colors[i]) for i in range(len(colors))]

    #drawing setup
    drawing = False
    prev_x, prev_y = None, None
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    draw_shape = False
    shape_on_screen = False
    circle_next = True
    tri_next = False
    rect_next = False

    last_draw_time = 0  # Tracks the last shape draw time
    draw_cooldown = 1  # Cooldown time in seconds (adjust as needed)

    mouth_open = False  # ✅ Prevents continuous sound spam

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)  # flips the image so movement appears correct
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #convert from BGR to RGB for mediapipe

        # detect hands
        hand_results = hand_recognition.detect_hands(rgb_image)
        # detect gestures
        gestures = gesture_recognition.recognize_gesture(rgb_image)
        #detect Face Mesh 
        face_mesh_results = face_recognition.detect_face_mesh(rgb_image)
        #check if Open Palm gesture is detected
        clear_screen = any(gesture == "Open_Palm" and score >= 0.5 for gesture, score in gestures)

        # reset canvas if Open Palm is detected
        if clear_screen:
            canvas[:] = 0
            prev_x, prev_y = None, None
            shape_on_screen = False

        # draw color selection boxes
        for x1, y1, x2, y2, color in color_boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)

        #checks if mout is open and plays sound
        if face_mesh_results.multi_face_landmarks:
            for face_landmarks in face_mesh_results.multi_face_landmarks:

                if is_mouth_open(face_landmarks) and not mouth_open:
                    mouth_open = True
                    pygame.mixer.music.play()

                elif not is_mouth_open(face_landmarks):
                    mouth_open = False  #reset when mouth closes

        # drawing functionality
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                x, y = int(index_finger_tip.x * 640), int(index_finger_tip.y * 480) #sets co-ordinates of fingertip
                Tx, Ty = int(thumb_tip.x * 640), int(thumb_tip.y * 480) #sets co-ordinates of thumb tip
                
                tri_pts = np.array([[Tx, Ty], [Tx - 30, Ty + 50], [Tx + 30, Ty + 50]], np.int32) #points for triangle
                tri_pts = tri_pts.reshape((-1, 1, 2))  # reshape to correct format      

                #checks if colour needs changing
                new_color = check_color_selection(x, y, color_boxes)
                if new_color:
                    selected_color = new_color

                #sets draw to true if finger up
                if is_pointing_up(hand_landmarks):
                    drawing = True
                else:
                    drawing = False
                    prev_x, prev_y = None, None

                #draws if finger up
                if drawing and prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), selected_color, 5)
                prev_x, prev_y = x, y

                #checks if thumb is down and there is not a shape on screen
                if is_thumb_down(hand_landmarks) and not shape_on_screen:
                    draw_shape = True
                
                else:
                    draw_shape = False

                #draws shape and applied cooldown so multiple are not drawn 
                current_time = time.time()
                if draw_shape and (current_time - last_draw_time > draw_cooldown):
                    if circle_next:
                        cv2.circle(canvas, (Tx, Ty), 100, selected_color, -1)
                        shape_on_screen = True
                        tri_next = True
                        circle_next = False  

                    elif tri_next:
                        tri_pts = np.array([[Tx, Ty], [Tx - 130, Ty + 150], [Tx + 130, Ty + 150]], np.int32)
                        tri_pts = tri_pts.reshape((-1, 1, 2))
                        cv2.fillPoly(canvas, [tri_pts], selected_color)
                        shape_on_screen = True
                        rect_next = True
                        tri_next = False  

                    elif rect_next:
                        cv2.rectangle(canvas, (Tx, Ty), (Tx + 140, Ty + 140), selected_color, -1)
                        shape_on_screen = True
                        circle_next = True  
                        rect_next = False  

                    last_draw_time = current_time  #adds cooldown

        #adds drawings on camera
        output = cv2.addWeighted(image, 1, canvas, 0.5, 0)

        gesture_text = "No Gesture Detected"
        if gestures:
            gesture_text = f"Gesture: {gestures[0][0]}"  # display first recognized gesture

        # displays gesture on screen
        cv2.putText(output, gesture_text, (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        #show everything
        cv2.imshow("Interface Programming", output)

        #close on esc press
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
