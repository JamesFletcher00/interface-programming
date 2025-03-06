import numpy as np
import cv2

bg = np.zeros((512,512,3), np.uint8)

#cv2.shape(background image, (x,y), size, BGR colour, thickness)
cv2.line(bg,(0,0),(511,511),(0,255,0),5)
cv2.circle(bg,(247,63), 64, (0,0,225), -1)

cv2.ellipse(bg,(256,256),(100,50),0,0,180,255,-1)

cv2.imshow('image',bg)
cv2.waitKey(0)
cv2.destroyAllWindows()