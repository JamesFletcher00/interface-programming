def is_fist(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    mcp_ids = [2, 5, 9, 13, 17]

    closed = 0
    for tip, mcp in zip(tips_ids[1:], mcp_ids[1:]):  
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[mcp].y:
            closed += 1

    return closed >= 3 

def is_ily_sign(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    for tip_id in tips:
        tip = hand_landmarks.landmark[tip_id]
        pip = hand_landmarks.landmark[tip_id - 2]
        fingers.append(tip.y < pip.y)  

    index_up = fingers[1]
    middle_down = not fingers[2]
    ring_down = not fingers[3]
    pinky_up = fingers[4]

    return index_up and middle_down and ring_down and pinky_up

def is_thumbs_down(hand_landmarks, img_height):
    lm = hand_landmarks.landmark

    thumb_tip_y = lm[4].y * img_height
    wrist_y = lm[0].y * img_height

    fingers_folded = all(
        lm[tip].y > lm[tip - 2].y for tip in [8, 12, 16, 20]
    )

    return thumb_tip_y > wrist_y and fingers_folded


def get_hand_center(hand_landmarks, frame_width, frame_height):
    x = int(hand_landmarks.landmark[9].x * frame_width)
    y = int(hand_landmarks.landmark[9].y * frame_height)
    return (x, y)
