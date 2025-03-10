import cv2
import numpy as np
import time

class Scene:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.bg = np.zeros((height, width, 3), dtype=np.uint8)
        self.draw_table()
        self.draw_net()  # Draw static elements once
    
    def draw_table(self):
        cv2.rectangle(self.bg, (150, 100), (500, 400), (0, 255, 0), -1)
        cv2.rectangle(self.bg, (150, 100), (500, 400), (255, 255, 255), 2)
    
    def draw_net(self):
        cv2.line(self.bg, (325, 100), (325, 400), (255, 255, 255), 2)
        start_x, end_x = 335, 300
        y_pairs = [
            (330, 360), (320, 350), (310, 340), (300, 330), (290, 320),
            (280, 310), (270, 300), (260, 290), (250, 280), (240, 270),
            (237, 260), (227, 250), (218, 240), (208, 230), (199, 220),
            (189, 210), (180, 200), (170, 190), (161, 180), (151, 170),
            (142, 160), (132, 150)
        ]

        for start_y, end_y in y_pairs:
            cv2.line(self.bg, (start_x, start_y), (end_x, end_y), (255, 255, 255), 1)
            if start_y in [237, 227, 218, 208, 199, 189, 180, 170, 161, 151, 142, 132]:
                start_x -= 1

class Ball:
    def __init__(self, x=250, y=250, radius=10):
        self.x = x
        self.y = y
        self.radius = radius
    
    def draw(self, frame):
        cv2.circle(frame, (self.x, self.y), self.radius, (0, 0, 255), -1)
    
    def move(self, dx, dy):
        self.x += dx
        self.y += dy

class Animation:
    def __init__(self):
        self.scene = Scene()
        self.ball = Ball()
    
    def animate(self):
        for _ in range(4):  # 4-frame animation
            frame = self.scene.bg.copy()  # Copy static background
            self.ball.draw(frame)
            cv2.imshow("Scene", frame)
            cv2.waitKey(200)  # Short delay
            self.ball.move(10, -5)  # Move the ball slightly
        
        cv2.destroyAllWindows()

animation = Animation()
animation.animate()
