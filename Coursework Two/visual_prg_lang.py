import cv2
import numpy as np
import mediapipe as mp
from code_blocks import ActionBlock, VariableBlock, ValueBlock, ControlBlock, OperatorBlock, StringBlock
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
# Setup
# ----------------------------

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)

success, img = cap.read()
if not success:
    exit("Failed to read from webcam.")

img_height, img_width, _ = img.shape
cv2.resizeWindow("Hand Tracking", 1280, 960)

PLAY_AREA_TOP = int(img_height * 0.3)

# ----------------------------
# UI State
# ----------------------------

menu_state = "main"
palette_blocks = []
live_blocks = []
dragged_block = None

# Buttons
button_labels = ["Functions", "Variables", "Values", "Strings", "Operators"]
buttons = []
button_width = img_width // len(button_labels)
button_height = 50

for i, label in enumerate(button_labels):
    x = i * button_width
    width = button_width if i < len(button_labels) - 1 else img_width - x
    buttons.append(Button(x, 0, width, button_height, label))

back_button = Button(10, img_height - 60, 100, 45, "Back", (180, 50, 50))

# ----------------------------
# Spawn Blocks by Category
# ----------------------------

def spawn_blocks_for_category(category):
    blocks = []
    spacing = 5
    x = 20
    y = PLAY_AREA_TOP - 70  # just above the play area

    if category == "Functions":
        for action in ["print"]:
            block = ActionBlock(x, y, action=action, value="x,y")
            blocks.append(block)
            x += block.width + spacing

    elif category == "Variables":
        for var in ["x", "y", "name"]:
            block = VariableBlock(x, y, name=var)
            blocks.append(block)
            x += block.width + spacing

    elif category == "Values":
        for val in range(10):
            block = ValueBlock(x, y, value=str(val))
            blocks.append(block)
            x += block.width + spacing

    elif category == "Strings":
        for text in ["Hello", "World", "James"]:
            block = ValueBlock(x, y, value=f'"{text}"')
            blocks.append(block)
            x += block.width + spacing

    elif category == "Operators":
        for op in ["+", "-", "x", "/", "="]:
            block = OperatorBlock(x, y, operator=op)
            blocks.append(block)
            x += block.width + spacing


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

    img_height, img_width, _ = flipped.shape
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
    # Draw UI
    # ----------------------------

    # Draw play area
    cv2.rectangle(flipped, (0, PLAY_AREA_TOP), (img_width - 1, img_height - 1), (0, 255, 0), 2)

    if menu_state == "main":
        for btn in buttons:
            btn.hovered = index_finger and btn.contains_point(*index_finger)
            btn.draw(flipped)

        if hand_open and index_finger:
            for btn in buttons:
                if btn.contains_point(*index_finger):
                    menu_state = btn.label
                    palette_blocks = spawn_blocks_for_category(menu_state)
                    break

    else:
        back_button.hovered = index_finger and back_button.contains_point(*index_finger)
        back_button.draw(flipped)

    if hand_open and index_finger and back_button.contains_point(*index_finger):
        palette_blocks = []
        menu_state = "main"
        dragged_block = None

    # Always draw live blocks (play area) and any dragged block
    for b in live_blocks:
        b.draw(flipped)
    if dragged_block:
        dragged_block.draw(flipped)

    # Only draw palette blocks in category view
    if menu_state != "main":
        for b in palette_blocks:
            b.draw(flipped)



    # ----------------------------
    # Block Interaction
    # ----------------------------

    def clone_block(block):
        if isinstance(block, ActionBlock):
            return ActionBlock(block.x, block.y, action=block.action, value=block.value)
        elif isinstance(block, VariableBlock):
            return VariableBlock(block.x, block.y, name=block.name)
        elif isinstance(block, ValueBlock):
            return ValueBlock(block.x, block.y, value=block.value)
        elif isinstance(block, StringBlock):
            return StringBlock(block.x, block.y, value=block.value)
        elif isinstance(block, ControlBlock):
            return ControlBlock(block.x, block.y, label=block.label)
        elif isinstance(block, OperatorBlock):
            return OperatorBlock(block.x, block.y, operator=block.operator)
        return None


    if index_finger:
        if hand_closed:
            if dragged_block is None:
                # Try to grab from palette
                for b in palette_blocks:
                    if b.contains_point(*index_finger):
                        dragged_block = clone_block(b)
                        break

                # If not found, try to grab from live blocks
                if dragged_block is None:
                    for b in live_blocks:
                        if b.contains_point(*index_finger):
                            dragged_block = b
                            live_blocks.remove(b)
                            break

            elif dragged_block:
                dragged_block.x = index_finger[0] - dragged_block.width // 2
                dragged_block.y = index_finger[1] - dragged_block.height // 2

        else:
            if dragged_block:
                if dragged_block.y > PLAY_AREA_TOP:
                    live_blocks.append(dragged_block)
                # Otherwise discard (only for palette clones)
                dragged_block = None


    # ----------------------------
    # Display Frame
    # ----------------------------

    cv2.imshow("Hand Tracking", flipped)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
