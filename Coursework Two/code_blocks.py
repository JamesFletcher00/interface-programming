import cv2
import numpy as np

def get_shape(stretch_gap=0):
    """
    Returns a blocky 'C'-like shape.
    `stretch_gap` controls vertical spacing between internal bands.
    """
    points = np.array([
        [400, 50],                 # top-right
        [400, 100],                # bend-in down
        [150, 100],                # inner-left top
        [150, 200 + stretch_gap],  # inner-left bottom (moved down if stretched)
        [350, 200 + stretch_gap],  # inner-right bottom
        [350, 250 + stretch_gap],  # lower-right outer
        [100, 250 + stretch_gap],  # bottom-left outer
        [100, 50]                  # return to start
    ], np.int32)

    return points.reshape((-1, 1, 2))

# Initialize window
cv2.namedWindow("Shape", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Shape", 600, 600)

# Toggle state
stretched = False

while True:
    # Create blank canvas
    img = np.zeros((512, 512, 3), dtype=np.uint8)

    # Draw the appropriate shape
    shape_points = get_shape(stretch_gap=100 if stretched else 0)
    cv2.fillPoly(img, [shape_points], color=(255, 255, 255))

    # Display result
    cv2.imshow("Shape", img)

    key = cv2.waitKey(10)
    if key == 27:  # ESC key exits
        break
    elif key == 32:  # SPACE toggles shape
        stretched = not stretched

cv2.destroyAllWindows()
