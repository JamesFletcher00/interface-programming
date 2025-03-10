import numpy as np
import cv2

grass_green = (32, 128, 2)
sky_blue = (225, 165, 55) 
white = (255,255,255)
brown = (15, 75, 130)
dark_brown = (33, 67, 101)
black = (0,0,0)

class Scene:
    def __init__(self, width=1200, height=600):
        self.bg = np.zeros((height, width, 3), np.uint8)
        self.elements = []

    def add_element(self, element):
        self.elements.append(element)

    def render(self):
        for element in self.elements:
            element.draw(self.bg)
        cv2.imshow('Football', self.bg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class Sky:
    def draw(self, bg):
        cv2.rectangle(bg, (0, 0), (1200, 300), sky_blue, -1)

class Cloud:
    def __init__(self, points, thickness=100):
        self.points = np.array(points, np.int32).reshape((-1, 1, 2))
        self.thickness = thickness

    def draw(self, bg):
        cv2.polylines(bg, [self.points], True, white, self.thickness)

class Pitch:
    def draw(self, bg):
        cv2.rectangle(bg, (0, 600), (1200, 300), grass_green, -1)
        cv2.line(bg, (0, 600), (200, 360), white, 5)
        cv2.line(bg, (1200, 600), (1000, 360), white, 5)
        cv2.line(bg, (0, 360), (1200, 360), white, 5)
        cv2.line(bg, (0, 598), (1200, 598), white, 5)

class Goal:
    def draw(self, bg):
        # Goalposts
        cv2.line(bg, (300, 355), (300, 120), white, 15)
        cv2.line(bg, (900, 355), (900, 120), white, 15)
        cv2.line(bg, (300, 120), (900, 120), white, 15)
        cv2.line(bg, (300, 120), (325, 140), white, 5)
        cv2.line(bg, (900, 120), (875, 140), white, 5)
        cv2.line(bg, (325, 140), (335, 330), white, 5)
        cv2.line(bg, (875, 140), (865, 330), white, 5)
        
        #net
        #center netting
        cv2.line(bg,(315,132), (885,132), white, 1)
        cv2.line(bg,(325,140), (875,140), white, 1)
        cv2.line(bg,(325,150), (875,150), white, 1)
        cv2.line(bg,(325,160), (875,160), white, 1)
        cv2.line(bg,(325,170), (875,170), white, 1)
        cv2.line(bg,(325,180), (875,180), white, 1)
        cv2.line(bg,(325,190), (875,190), white, 1)
        cv2.line(bg,(325,200), (875,200), white, 1)
        cv2.line(bg,(325,210), (875,210), white, 1)
        cv2.line(bg,(330,220), (870,220), white, 1)
        cv2.line(bg,(330,230), (870,230), white, 1)
        cv2.line(bg,(330,240), (870,240), white, 1)
        cv2.line(bg,(335,250), (865,250), white, 1)
        cv2.line(bg,(335,260), (865,260), white, 1)
        cv2.line(bg,(335,270), (865,270), white, 1)
        cv2.line(bg,(335,280), (865,280), white, 1)
        cv2.line(bg,(335,290), (865,290), white, 1)
        cv2.line(bg,(335,300), (865,300), white, 1)
        cv2.line(bg,(335,310), (865,310), white, 1)
        cv2.line(bg,(335,320), (865,320), white, 1)
        cv2.line(bg,(335,330), (865,330), white, 1)

        #right side netting
        cv2.line(bg,(865,330), (905,360), white, 1)
        cv2.line(bg,(865,320), (905,350), white, 1)
        cv2.line(bg,(865,310), (905,340), white, 1)
        cv2.line(bg,(865,300), (905,330), white, 1)
        cv2.line(bg,(865,290), (905,320), white, 1)
        cv2.line(bg,(865,280), (905,310), white, 1)
        cv2.line(bg,(865,270), (905,300), white, 1)
        cv2.line(bg,(865,260), (905,290), white, 1)
        cv2.line(bg,(865,250), (905,280), white, 1)
        cv2.line(bg,(865,240), (905,270), white, 1)
        cv2.line(bg,(874,237), (905,260), white, 1)
        cv2.line(bg,(874,227), (905,250), white, 1)
        cv2.line(bg,(875,218), (905,240), white, 1)
        cv2.line(bg,(875,208), (905,230), white, 1)
        cv2.line(bg,(876,199), (905,220), white, 1)
        cv2.line(bg,(876,189), (905,210), white, 1)
        cv2.line(bg,(877,180), (905,200), white, 1)
        cv2.line(bg,(877,170), (905,190), white, 1)
        cv2.line(bg,(878,161), (905,180), white, 1)
        cv2.line(bg,(878,151), (905,170), white, 1)
        cv2.line(bg,(880,142), (905,160), white, 1)
        cv2.line(bg,(880,132), (905,150), white, 1)

        #left side netting
        cv2.line(bg,(335,330), (300,360), white, 1)
        cv2.line(bg,(335,320), (300,350), white, 1)
        cv2.line(bg,(335,310), (300,340), white, 1)
        cv2.line(bg,(335,300), (300,330), white, 1)
        cv2.line(bg,(335,290), (300,320), white, 1)
        cv2.line(bg,(335,280), (300,310), white, 1)
        cv2.line(bg,(335,270), (300,300), white, 1)
        cv2.line(bg,(335,260), (300,290), white, 1)
        cv2.line(bg,(335,250), (300,280), white, 1)
        cv2.line(bg,(335,240), (300,270), white, 1)
        cv2.line(bg,(326,237), (300,260), white, 1)
        cv2.line(bg,(326,227), (300,250), white, 1)
        cv2.line(bg,(325,218), (300,240), white, 1)
        cv2.line(bg,(325,208), (300,230), white, 1)
        cv2.line(bg,(324,199), (300,220), white, 1)
        cv2.line(bg,(324,189), (300,210), white, 1)
        cv2.line(bg,(323,180), (300,200), white, 1)
        cv2.line(bg,(323,170), (300,190), white, 1)
        cv2.line(bg,(322,161), (300,180), white, 1)
        cv2.line(bg,(322,151), (300,170), white, 1)
        cv2.line(bg,(320,142), (300,160), white, 1)
        cv2.line(bg,(320,132), (300,150), white, 1)

        #horizontal netting
        cv2.line(bg,(335,120), (350,330), white, 1)
        cv2.line(bg,(345,120), (360,330), white, 1)
        cv2.line(bg,(355,120), (370,330), white, 1)
        cv2.line(bg,(365,120), (380,330), white, 1)
        cv2.line(bg,(375,120), (390,330), white, 1)
        cv2.line(bg,(385,120), (400,330), white, 1)
        cv2.line(bg,(395,120), (410,330), white, 1)
        cv2.line(bg,(405,120), (420,330), white, 1)
        cv2.line(bg,(415,120), (430,330), white, 1)
        cv2.line(bg,(425,120), (440,330), white, 1)
        cv2.line(bg,(435,120), (450,330), white, 1)
        cv2.line(bg,(445,120), (460,330), white, 1)
        cv2.line(bg,(455,120), (470,330), white, 1)
        cv2.line(bg,(465,120), (480,330), white, 1)
        cv2.line(bg,(475,120), (490,330), white, 1)
        cv2.line(bg,(485,120), (500,330), white, 1)
        cv2.line(bg,(495,120), (510,330), white, 1)
        cv2.line(bg,(505,120), (520,330), white, 1)
        cv2.line(bg,(515,120), (530,330), white, 1)
        cv2.line(bg,(525,120), (540,330), white, 1)
        cv2.line(bg,(535,120), (550,330), white, 1)
        cv2.line(bg,(545,120), (560,330), white, 1)
        cv2.line(bg,(555,120), (570,330), white, 1)
        cv2.line(bg,(565,120), (580,330), white, 1)
        cv2.line(bg,(575,120), (590,330), white, 1)
        cv2.line(bg,(590,120), (600,330), white, 1)

        cv2.line(bg,(865,120), (850,330), white, 1)
        cv2.line(bg,(855,120), (840,330), white, 1)
        cv2.line(bg,(845,120), (830,330), white, 1)
        cv2.line(bg,(835,120), (820,330), white, 1)
        cv2.line(bg,(825,120), (810,330), white, 1)
        cv2.line(bg,(815,120), (800,330), white, 1)
        cv2.line(bg,(805,120), (790,330), white, 1)
        cv2.line(bg,(795,120), (780,330), white, 1)
        cv2.line(bg,(785,120), (770,330), white, 1)
        cv2.line(bg,(775,120), (760,330), white, 1)
        cv2.line(bg,(765,120), (750,330), white, 1)
        cv2.line(bg,(755,120), (740,330), white, 1)
        cv2.line(bg,(745,120), (730,330), white, 1)
        cv2.line(bg,(735,120), (720,330), white, 1)
        cv2.line(bg,(725,120), (710,330), white, 1)
        cv2.line(bg,(715,120), (700,330), white, 1)
        cv2.line(bg,(705,120), (690,330), white, 1)
        cv2.line(bg,(695,120), (680,330), white, 1)
        cv2.line(bg,(685,120), (670,330), white, 1)
        cv2.line(bg,(675,120), (660,330), white, 1)
        cv2.line(bg,(665,120), (650,330), white, 1)
        cv2.line(bg,(655,120), (640,330), white, 1)
        cv2.line(bg,(645,120), (630,330), white, 1)
        cv2.line(bg,(635,120), (620,330), white, 1)
        cv2.line(bg,(625,120), (610,330), white, 1)
        cv2.line(bg,(610,120), (600,330), white, 1)

        cv2.line(bg,(885,120), (875,336), white, 1)
        cv2.line(bg,(890,120), (885,343), white, 1)

        cv2.line(bg,(315,120), (325,338), white, 1)
        cv2.line(bg,(310,120), (315,346), white, 1)

"""
        start_Lx, end_Lx = 335, 300
        start_Rx, end_Rx = 865, 905
        y_pairs = [
            (330, 360), (320, 350), (310, 340), (300, 330), (290, 320),
            (280, 310), (270, 300), (260, 290), (250, 280), (240, 270),
            (237, 260), (227, 250), (218, 240), (208, 230), (199, 220),
            (189, 210), (180, 200), (170, 190), (161, 180), (151, 170),
            (142, 160), (132, 150)
        ]
        for start_y, end_y in y_pairs:
            cv2.line(bg, (start_Lx, start_y), (end_Lx, end_y), white, 1)
            if start_y in [237, 227, 218, 208, 199, 189, 180, 170, 161, 151, 142, 132]:
                start_Lx -= 1
        for start_y, end_y in y_pairs:
            cv2.line(bg, (start_Rx, start_y), (end_Rx, end_y), white, 1)
            if start_y in [237, 227, 218, 208, 199, 189, 180, 170, 161, 151, 142, 132]:
                start_Rx -= 1
"""


class Name:
    def draw(self, bg):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(bg,' - - - - - - James Fletcher - - - - - - ',(413,125), font, 0.5,black,1,cv2.LINE_AA)

class Ball:
    def draw(self, bg):
        cv2.ellipse(bg, (350, 163), (22, 28), -30, 0, 360, white, -1)
        cv2.ellipse(bg, (350, 163), (22, 28), -30, 0, 360, black, 1)

class DirtPatch:
    def __init__(self, points, color, thickness):
        self.points = np.array(points, np.int32).reshape((-1, 1, 2))
        self.color = color
        self.thickness = thickness

    def draw(self, bg):
        cv2.polylines(bg, [self.points], True, self.color, self.thickness)

# Initialize scene
scene = Scene()

# Add elements to the scene
scene.add_element(Sky())
scene.add_element(Cloud([[30, 80], [50, 50], [80, 80], [70, 70]], 100))
scene.add_element(Cloud([[290, 80], [320, 50], [380, 80], [340, 70], [375, 50]], 60))
scene.add_element(Cloud([[1000, 80], [1020, 50], [1080, 80], [1040, 70], [1075, 50]], 100))
scene.add_element(Pitch())
scene.add_element(Goal())
scene.add_element(Ball())
scene.add_element(Name())
scene.add_element(DirtPatch([[570, 450], [610, 500], [670, 480], [650, 470]], dark_brown, 100))
scene.add_element(DirtPatch([[620, 470], [650, 500], [670, 480], [650, 470]], brown, 30))

# Render the scene
scene.render()
