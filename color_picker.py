import cv2

class ColorPicker:
    def __init__(self, frame_width, height=70):
        self.colors = [
            ("red", (0, 0, 255)),
            ("blue", (255, 0, 0)),
            ("green", (0, 255, 0)),
            ("yellow", (0, 255, 255)),
            ("purple", (255, 0, 255)),
            ("white", (255, 255, 255)),
            ("eraser", (0, 0, 0)),
        ]
        self.height = height
        self.box_w = frame_width // len(self.colors)
        self.selected = 0

    @property
    def color(self):
        return self.colors[self.selected][1]

    @property
    def thickness(self):
        return 60 if self.colors[self.selected][0] == "eraser" else 6

    def select(self, x, y):
        """If (x, y) is inside the palette, pick that color. Returns True if inside."""
        if y < self.height:
            self.selected = min(x // self.box_w, len(self.colors) - 1)
            return True
        return False

    def draw(self, frame):
        for i, (label, color) in enumerate(self.colors):
            x1, x2 = i * self.box_w, (i + 1) * self.box_w
            cv2.rectangle(frame, (x1, 0), (x2, self.height), color, cv2.FILLED)
            cv2.rectangle(frame, (x1, 0), (x2, self.height), (120, 120, 120), 1)
            if label == "eraser":
                text = "ERASER"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2

                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

                text_x = x1 + (self.box_w - text_size[0]) // 2
                text_y = (self.height + text_size[1]) // 2

                cv2.putText(frame, text, (text_x, text_y),
                            font, font_scale, (255, 255, 255), thickness)
                
        x1, x2 = self.selected * self.box_w, (self.selected + 1) * self.box_w
        cv2.rectangle(frame, (x1, 0), (x2, self.height), (255, 255, 255), 3)