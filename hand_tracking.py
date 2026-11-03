import math
import cv2
import mediapipe as mp


class HandTracker:
    def __init__(
        self,
        max_hands=1,
        detection_conf=0.5,
        tracking_conf=0.5,
        smoothing=0.65
    ):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
            model_complexity=1
        )

        self.results = None
        self.smoothing = smoothing
        self.previous_landmarks = None

    def find_hands(self, frame, draw=True):
        """Detect hands and draw their skeleton."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)

        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand,
                        self.mp_hands.HAND_CONNECTIONS
                    )
        else:
            # Reset smoothing when tracking is lost.
            self.previous_landmarks = None

        return frame

    def get_landmarks(self, frame, hand_no=0):
        """Return smoothed hand landmarks in pixel coordinates."""
        landmarks = []

        if (
            self.results
            and self.results.multi_hand_landmarks
            and hand_no < len(self.results.multi_hand_landmarks)
        ):
            hand = self.results.multi_hand_landmarks[hand_no]
            h, w, _ = frame.shape

            current_landmarks = []

            for i, lm in enumerate(hand.landmark):
                x = lm.x * w
                y = lm.y * h

                current_landmarks.append([i, x, y])

            # Apply exponential smoothing to reduce jitter.
            if self.previous_landmarks is not None:
                smoothed_landmarks = []

                for current, previous in zip(
                    current_landmarks, self.previous_landmarks
                ):
                    x = (
                        self.smoothing * current[1]
                        + (1 - self.smoothing) * previous[1]
                    )
                    y = (
                        self.smoothing * current[2]
                        + (1 - self.smoothing) * previous[2]
                    )

                    smoothed_landmarks.append([current[0], x, y])

                current_landmarks = smoothed_landmarks

            self.previous_landmarks = current_landmarks

            landmarks = [
                [i, int(x), int(y)]
                for i, x, y in current_landmarks
            ]

        return landmarks

    def fingers_up(self, lm):
        """Return [thumb, index, middle, ring, pinky]."""
        if not lm:
            return []

        def dist(a, b):
            return math.hypot(
                lm[a][1] - lm[b][1],
                lm[a][2] - lm[b][2]
            )

        fingers = []
        fingers.append(1 if dist(4, 17) > dist(3, 17) else 0)

        for tip in [8, 12, 16, 20]:
            fingers.append(
                1 if lm[tip][2] < lm[tip - 2][2] else 0
            )

        return fingers

    def distance(self, lm, a, b):
        """Pixel distance between two landmarks."""
        return math.hypot(
            lm[a][1] - lm[b][1],
            lm[a][2] - lm[b][2]
        )
