import cv2
import numpy as np
import mediapipe as mp
from code_blocks import ActionBlock, VariableBlock, ValueBlock, ControlBlock
from gestures import is_fist, get_hand_center

# ----------------------------
# UI Classes
# ----------------------------

class Button:
    def __init__(self, x, y, width, height, label, color=(100, 100, 255)):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label
        self.color = color
        self.hovered = False

    def draw(self, frame):
        color = (200, 200, 100) if self.hovered else self.color
        cv2.rectangle(frame, (self.x, self.y),
                      (self.x + self.width, self.y + self.height), color, -1)
        cv2.rectangle(frame, (self.x, self.y),
                      (self.x + self.width, self.y + self.height), (0, 0, 0), 2)

        font_scale = 0.6
        thickness = 2
        text_size = cv2.getTextSize(self.label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = self.x + (self.width - text_size[0]) // 2
        text_y = self.y + (self.height + text_size[1]) // 2
        cv2.putText(frame, self.label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    def contains_point(self, px, py):
        return self.x <= px <= self.x + self.width and self.y <= py <= self.y + self.height

# ----------------------------
# MediaPipe Setup
# ----------------------------

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)

# Read first frame
success, img = cap.read()
if not success:
    exit("Failed to read from webcam.")

img_height, img_width, _ = img.shape
cv2.resizeWindow("Hand Tracking", img_width, img_height)
PLAY_AREA_TOP = int(img_height * 0.3)

# ----------------------------
# Menu State Setup
# ----------------------------

menu_state = "main"
spawned_blocks = []
selected_block = None

# Back Button
back_button = Button(10, 10, 80, 40, "Back", (180, 50, 50))

# Main menu buttons
button_labels = ["Actions", "Variables", "Values", "Strings", "Control"]
buttons = []
button_width = img_width // len(button_labels)
button_height = 50

for i, label in enumerate(button_labels):
    x = i * button_width
    width = button_width if i < len(button_labels) - 1 else img_width - x
    buttons.append(Button(x, 0, width, button_height, label))

# ----------------------------
# Block Factory
# ----------------------------

def spawn_blocks_for_category(category):
    blocks = []
    x, y = 50, 100
    spacing = 10

    if category == "Actions":
        for action in ["print", "add", "multiply", "subtract", "divide"]:
            blocks.append(ActionBlock(x, y, action=action, value="x,y"))
            y += 70 + spacing

    elif category == "Variables":
        for var in ["x", "y", "name"]:
            blocks.append(VariableBlock(x, y, name=var))
            y += 60 + spacing

    elif category == "Values":
        for val in range(10):
            blocks.append(ValueBlock(x, y, value=str(val)))
            y += 50 + spacing

    elif category == "Strings":
        for text in ["Hello", "World", "James"]:
            blocks.append(ValueBlock(x, y, value=f'"{text}"'))
            y += 50 + spacing

    elif category == "Control":
        blocks.append(ControlBlock(x, y, label="Repeat"))

    return blocks

# ----------------------------
# Main Loop
# ----------------------------

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    flipped = cv2.flip(img, 1)

    hand_center = None
    index_finger = None
    hand_closed = False
    hand_open = True

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            for lm in hand_lms.landmark:
                cx = img_width - int(lm.x * img_width)
                cy = int(lm.y * img_height)
                cv2.circle(flipped, (cx, cy), 4, (0, 0, 255), -1)

            raw_x, raw_y = get_hand_center(hand_lms, img_width, img_height)
            flipped_x = img_width - raw_x
            hand_center = (flipped_x, raw_y)

            index_x = img_width - int(hand_lms.landmark[8].x * img_width)
            index_y = int(hand_lms.landmark[8].y * img_height)
            index_finger = (index_x, index_y)

            hand_closed = is_fist(hand_lms)
            hand_open = not hand_closed
            break

    # ----------------------------
    # State: Main Menu
    # ----------------------------
    if menu_state == "main":
        for btn in buttons:
            btn.hovered = index_finger and btn.contains_point(*index_finger)
            btn.draw(flipped)

        if hand_open and index_finger:
            for btn in buttons:
                if btn.contains_point(*index_finger):
                    menu_state = btn.label
                    spawned_blocks = spawn_blocks_for_category(btn.label)
                    break

    # ----------------------------
    # State: In Category
    # ----------------------------
    else:
        back_button.hovered = index_finger and back_button.contains_point(*index_finger)
        back_button.draw(flipped)

        if hand_open and index_finger and back_button.contains_point(*index_finger):
            # Keep only blocks below play area
            spawned_blocks = [b for b in spawned_blocks if b.y > PLAY_AREA_TOP]
            menu_state = "main"

        for block in spawned_blocks:
            block.draw(flipped)

    # ----------------------------
    # Play Area & Block Movement
    # ----------------------------

    cv2.rectangle(flipped, (0, PLAY_AREA_TOP), (img_width - 1, img_height - 1), (0, 255, 0), 2)

    if index_finger:
        if hand_closed:
            if selected_block is None:
                for block in spawned_blocks:
                    if block.contains_point(*index_finger):
                        selected_block = block
                        break
            elif selected_block:
                selected_block.x = index_finger[0] - selected_block.width // 2
                selected_block.y = index_finger[1] - selected_block.height // 2
        else:
            selected_block = None

    # ----------------------------
    # Show Frame
    # ----------------------------

    cv2.imshow("Hand Tracking", flipped)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
