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
        self.children = []  # ✅ new: stackable child blocks
        label = action
        super().__init__(x, y, 160, 60, label, (255, 153, 51))  # Orange

    def draw(self, canvas):
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

        # Draw label (e.g. print)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, self.label, (x + 10, y + 25),
                    font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        # Draw all children stacked inside
        for i, child in enumerate(self.children):
            child.draw(canvas)

    def move(self, dx, dy):
        self.x += dx
        self.y += dy
        for child in self.children:
            child.move(dx, dy)

    def get_snap_zone(self):
        # Return the snapping rectangle inside the C-hole
        x = self.x + 30
        y = self.y + 30 + len(self.children) * 50
        width = self.width - 60
        height = 40
        return (x, y, width, height)
    
    def generate_code(self):
        if self.action == "repeat":
            # Expect first child to be a value (how many times)
            if not self.children:
                return "# repeat with no body"

            count_expr = "1"
            body_blocks = self.children

            # If first child is a ValueBlock or OperatorBlock, treat as count
            if isinstance(body_blocks[0], (ValueBlock, OperatorBlock)):
                count_expr = body_blocks[0].generate_code()
                body_blocks = body_blocks[1:]

            body_code = "\n    ".join(
                b.generate_code() for b in body_blocks if hasattr(b, "generate_code")
            )
            return f"for _ in range({count_expr}):\n    {body_code}"

        # Default behavior (e.g. print)
        args = [child.generate_code() for child in self.children]
        return f"{self.action}({', '.join(args)})"




class VariableBlock(Block):
    def __init__(self, x, y, name="x"):
        self.name = name
        label = name
        super().__init__(x, y, 100, 45, label, (0, 200, 0))  # Green

    def draw(self, canvas):
        x, y, w, h = self.x, self.y, self.width, self.height
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.color, -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, self.label, (x + 10, y + h // 2 + 5),
                    font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        
    def generate_code(self):
        return self.name




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
        
    def generate_code(self):
        return str(self.value)


class StringBlock(ValueBlock):
    def __init__(self, x, y, value='"Hello"'):
        super().__init__(x, y, value)
        self.color = (0, 180, 255)  # Light blue

    def generate_code(self):
        return str(self.value)


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
        self.left_child = None
        self.right_child = None
        label = operator
        super().__init__(x, y, 140, 50, label, (102, 102, 255))  # Blue

    def draw(self, canvas):
        x, y, w, h = self.x, self.y, self.width, self.height
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self.color, -1)

        # Socket positions
        socket_size = 25
        padding = 10
        left_socket = (x + padding, y + (h - socket_size) // 2, socket_size, socket_size)
        right_socket = (x + w - padding - socket_size, y + (h - socket_size) // 2, socket_size, socket_size)

        # Draw sockets or child blocks
        if self.left_child:
            self.left_child.draw(canvas)
        else:
            cv2.rectangle(canvas, (left_socket[0], left_socket[1]),
                          (left_socket[0] + socket_size, left_socket[1] + socket_size),
                          (255, 255, 255), -1)

        if self.right_child:
            self.right_child.draw(canvas)
        else:
            cv2.rectangle(canvas, (right_socket[0], right_socket[1]),
                          (right_socket[0] + socket_size, right_socket[1] + socket_size),
                          (255, 255, 255), -1)

        # Draw operator symbol
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(self.label, font, 0.6, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(canvas, self.label, (text_x, text_y),
                    font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    def move(self, dx, dy):
        self.x += dx
        self.y += dy
        if self.left_child:
            self.left_child.move(dx, dy)
        if self.right_child:
            self.right_child.move(dx, dy)

    def get_socket_rects(self):
        # Return the left and right socket positions
        socket_size = 25
        padding = 10
        h = self.height
        left = (self.x + padding, self.y + (h - socket_size) // 2, socket_size, socket_size)
        right = (self.x + self.width - padding - socket_size, self.y + (h - socket_size) // 2, socket_size, socket_size)
        return left, right

    def generate_code(self):
        left = self.left_child.generate_code() if self.left_child else "?"
        right = self.right_child.generate_code() if self.right_child else "?"
        return f"({left} {self.operator} {right})"
