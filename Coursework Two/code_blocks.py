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

    def draw(self, canvas):
        pts = self._get_outline()
        cv2.fillPoly(canvas, [pts], self.color)

        # Draw text label
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, self.label, (self.x + 10, self.y + self.height // 2),
                    font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def _get_outline(self):
        # Create puzzle-shaped block with optional connectors
        path = []
        x, y, w, h = self.x, self.y, self.width, self.height

        # Top edge
        path.append([x, y])
        if self.input_conn:
            path += [
                [x + w//3, y],
                [x + w//3 + 10, y + 10],
                [x + 2*w//3 - 10, y + 10],
                [x + 2*w//3, y]
            ]
        path.append([x + w, y])

        # Right edge
        path.append([x + w, y + h])

        # Bottom edge with connector if output
        if self.output_conn:
            path += [
                [x + 2*w//3, y + h],
                [x + 2*w//3 - 10, y + h + 10],
                [x + w//3 + 10, y + h + 10],
                [x + w//3, y + h]
            ]
        path.append([x, y + h])

        # Left edge
        path.append([x, y])

        return np.array(path, np.int32).reshape((-1, 1, 2))

    def contains_point(self, px, py):
        return self.x <= px <= self.x + self.width and self.y <= py <= self.y + self.height

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

class ActionBlock(Block):
    def __init__(self, x, y, action="print", value="hello"):
        label = f"{action}(\"{value}\")"
        super().__init__(x, y, 160, 60, label, (255, 153, 51), input_conn=False, output_conn=True)
        self.action = action
        self.value = value

class VariableBlock(Block):
    def __init__(self, x, y, name="x"):
        label = f"Var: {name}"
        super().__init__(x, y, 140, 50, label, (102, 204, 0), input_conn=True, output_conn=True)
        self.name = name

class ValueBlock(Block):
    def __init__(self, x, y, value="5"):
        label = f"Val: {value}"
        super().__init__(x, y, 100, 40, label, (200, 200, 0), input_conn=False, output_conn=False)
        self.value = value

class ControlBlock(Block):
    def __init__(self, x, y, label="Repeat"):
        super().__init__(x, y, 180, 100, label, (51, 153, 255), input_conn=True, output_conn=True)
        self.inner_padding = 20
        self.children = []

    def draw(self, canvas):
        pts = np.array([
            [self.x, self.y],
            [self.x + self.width, self.y],
            [self.x + self.width, self.y + 30],
            [self.x + 80, self.y + 30],
            [self.x + 80, self.y + self.height - 30],
            [self.x + self.width, self.y + self.height - 30],
            [self.x + self.width, self.y + self.height],
            [self.x, self.y + self.height],
            [self.x, self.y]
        ], np.int32)
        cv2.fillPoly(canvas, [pts.reshape((-1, 1, 2))], self.color)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, self.label, (self.x + 10, self.y + 25),
                    font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        for child in self.children:
            child.draw(canvas)

    def add_child(self, block):
        self.children.append(block)
