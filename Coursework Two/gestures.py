def is_fist(hand_landmarks):
    # Simple fist detection based on finger tip and knuckle y-positions
    tips_ids = [4, 8, 12, 16, 20]
    mcp_ids = [2, 5, 9, 13, 17]

    closed = 0
    for tip, mcp in zip(tips_ids[1:], mcp_ids[1:]):  # skip thumb for simplicity
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[mcp].y:
            closed += 1

    return closed >= 3  # majority of fingers folded

def get_hand_center(hand_landmarks, frame_width, frame_height):
    x = int(hand_landmarks.landmark[9].x * frame_width)
    y = int(hand_landmarks.landmark[9].y * frame_height)
    return (x, y)
