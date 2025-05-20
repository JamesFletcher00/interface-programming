import cv2
import numpy as np

class Block:
    def __init__(self, x, y, width, height, label, color, input_conn=False, output_conn=False):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label
        self.color = color
        self.selected = False
        self.input_conn = input_conn
        self.output_conn = output_conn

    def contains_point(self, px, py):
        return self.x <= px <= self.x + self.width and self.y <= py <= self.y + self.height

    def move(self, dx, dy):
        self.x += dx
        self.y += dy


class ActionBlock(Block):
    def __init__(self, x, y, action="print", value=""):
        self.action = action
        self.value = value
        label = action
        super().__init__(x, y, 160, 60, label, (255, 153, 51))  # Orange

    def draw(self, canvas):
        # Draw U-shape block
        x, y, w, h = self.x, self.y, self.width, self.height
        mid_cut = 30

        pts = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h // 3],
            [x + mid_cut, y + h // 3],
            [x + mid_cut, y + h * 2 // 3],
            [x + w, y + h * 2 // 3],
            [x + w, y + h],
            [x, y + h],
            [x, y]
        ], np.int32)
        cv2.fillPoly(canvas, [pts.reshape((-1, 1, 2))], self.color)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, self.label, (x + 10, y + 25),
                    font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


class VariableBlock(Block):
    def __init__(self, x, y, name="x"):
        self.name = name
        label = f"{name} ="
        super().__init__(x, y, 140, 50, label, (0, 200, 0))  # Green

    def draw(self, canvas):
        x, y, w, h = self.x, self.y, self.width, self.height
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.color, -1)

        # Draw label
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, self.label, (x + 10, y + h // 2 + 5),
                    font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        # Draw white socket on the right
        socket_w = 30
        socket_h = 30
        sx = x + w - socket_w - 10
        sy = y + (h - socket_h) // 2
        cv2.rectangle(canvas, (sx, sy), (sx + socket_w, sy + socket_h), (255, 255, 255), -1)


class ValueBlock(Block):
    def __init__(self, x, y, value="5"):
        label = f"{value}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1

        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        width = text_size[0] + 12
        height = text_size[1] + 14

        super().__init__(x, y, width, height, label, (160, 64, 160))  # Purple
        self.value = value

    def draw(self, canvas):
        x, y, w, h = self.x, self.y, self.width, self.height
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.color, -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        text_size = cv2.getTextSize(self.label, font, font_scale, thickness)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2

        cv2.putText(canvas, self.label, (text_x, text_y),
                    font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


class StringBlock(ValueBlock):
    def __init__(self, x, y, value='"Hello"'):
        super().__init__(x, y, value)
        self.color = (0, 180, 255)  # Light blue


class ControlBlock(Block):
    def __init__(self, x, y, label="Repeat"):
        super().__init__(x, y, 180, 80, label, (255, 153, 51))  # Same color as Action
        self.children = []

    def draw(self, canvas):
        x, y, w, h = self.x, self.y, self.width, self.height
        mid_cut = 50

        pts = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + 25],
            [x + mid_cut, y + 25],
            [x + mid_cut, y + h - 25],
            [x + w, y + h - 25],
            [x + w, y + h],
            [x, y + h],
            [x, y]
        ], np.int32)
        cv2.fillPoly(canvas, [pts.reshape((-1, 1, 2))], self.color)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, self.label, (x + 10, y + 20),
                    font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        for child in self.children:
            child.draw(canvas)


class OperatorBlock(Block):
    def __init__(self, x, y, operator="+"):
        self.operator = operator
        label = operator
        super().__init__(x, y, 140, 50, label, (102, 102, 255))  # Blue

    def draw(self, canvas):
        x, y, w, h = self.x, self.y, self.width, self.height
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.color, -1)

        # Draw sockets
        socket_size = 25
        padding = 10

        # Left socket
        cv2.rectangle(canvas, (x + padding, y + (h - socket_size) // 2),
                      (x + padding + socket_size, y + (h + socket_size) // 2),
                      (255, 255, 255), -1)

        # Right socket
        cv2.rectangle(canvas, (x + w - padding - socket_size, y + (h - socket_size) // 2),
                      (x + w - padding, y + (h + socket_size) // 2),
                      (255, 255, 255), -1)

        # Center operator symbol
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(self.label, font, 0.6, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(canvas, self.label, (text_x, text_y),
                    font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
