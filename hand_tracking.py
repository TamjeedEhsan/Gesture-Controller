# import math
# import cv2
# import mediapipe as mp

# class HandTracker:
#     def __init__(
#         self,
#         max_hands=1,
#         detection_conf=0.5,
#         tracking_conf=0.5,
#         smoothing=0.65
#     ):
#         self.mp_hands = mp.solutions.hands
#         self.mp_draw = mp.solutions.drawing_utils

#         self.hands = self.mp_hands.Hands(
#             max_num_hands=max_hands,
#             min_detection_confidence=detection_conf,
#             min_tracking_confidence=tracking_conf,
#             model_complexity=1
#         )

#         self.results = None
#         self.smoothing = smoothing
#         self.previous_landmarks = None

#     def find_hands(self, frame, draw=True):
#         """Detect hands and draw their skeleton."""
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         self.results = self.hands.process(rgb)

#         if self.results.multi_hand_landmarks:
#             for hand in self.results.multi_hand_landmarks:
#                 if draw:
#                     self.mp_draw.draw_landmarks(
#                         frame,
#                         hand,
#                         self.mp_hands.HAND_CONNECTIONS
#                     )
#         else:
#             # Reset smoothing when tracking is lost.
#             self.previous_landmarks = None

#         return frame

#     def get_landmarks(self, frame, hand_no=0):
#         """Return smoothed hand landmarks in pixel coordinates."""
#         landmarks = []

#         if (
#             self.results
#             and self.results.multi_hand_landmarks
#             and hand_no < len(self.results.multi_hand_landmarks)
#         ):
#             hand = self.results.multi_hand_landmarks[hand_no]
#             h, w, _ = frame.shape

#             current_landmarks = []

#             for i, lm in enumerate(hand.landmark):
#                 x = lm.x * w
#                 y = lm.y * h

#                 current_landmarks.append([i, x, y])

#             # Apply exponential smoothing to reduce jitter.
#             if self.previous_landmarks is not None:
#                 smoothed_landmarks = []

#                 for current, previous in zip(
#                     current_landmarks, self.previous_landmarks
#                 ):
#                     x = (
#                         self.smoothing * current[1]
#                         + (1 - self.smoothing) * previous[1]
#                     )
#                     y = (
#                         self.smoothing * current[2]
#                         + (1 - self.smoothing) * previous[2]
#                     )

#                     smoothed_landmarks.append([current[0], x, y])

#                 current_landmarks = smoothed_landmarks

#             self.previous_landmarks = current_landmarks

#             landmarks = [
#                 [i, int(x), int(y)]
#                 for i, x, y in current_landmarks
#             ]

#         return landmarks

#     def fingers_up(self, lm):
#         """Return [thumb, index, middle, ring, pinky]."""
#         if not lm:
#             return []

#         def dist(a, b):
#             return math.hypot(
#                 lm[a][1] - lm[b][1],
#                 lm[a][2] - lm[b][2]
#             )

#         fingers = []
#         fingers.append(1 if dist(4, 17) > dist(3, 17) else 0)

#         for tip in [8, 12, 16, 20]:
#             fingers.append(
#                 1 if lm[tip][2] < lm[tip - 2][2] else 0
#             )

#         return fingers

#     def distance(self, lm, a, b):
#         """Pixel distance between two landmarks."""
#         return math.hypot(
#             lm[a][1] - lm[b][1],
#             lm[a][2] - lm[b][2]
#         )








import math
import cv2
import mediapipe as mp


class HandTracker:
    def __init__(
        self,
        max_hands=1,
        detection_conf=0.5,
        tracking_conf=0.5,
        still_mix=0.22,
    ):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
            model_complexity=0
        )

        self.results = None
        self.still_mix = still_mix
        self.previous_landmarks = None
        self.previous_fingers = None
        self.missed_frames = 0

    def find_hands(self, frame, draw=True):
        """Detect hands and draw their skeleton."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)

        if self.results.multi_hand_landmarks:
            self.missed_frames = 0
            if draw:
                for hand in self.results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand,
                        self.mp_hands.HAND_CONNECTIONS
                    )
        else:
            self.missed_frames += 1
            if self.missed_frames >= 6:
                self.previous_landmarks = None
                self.previous_fingers = None

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
                current_landmarks.append([i, lm.x * w, lm.y * h])

            palm = math.hypot(
                current_landmarks[0][1] - current_landmarks[9][1],
                current_landmarks[0][2] - current_landmarks[9][2]
            )
            palm = max(palm, 40.0)

            if self.previous_landmarks is not None:
                smoothed_landmarks = []
                for current, previous in zip(current_landmarks, self.previous_landmarks):
                    x, y = self._blend(
                        current[1], current[2],
                        previous[1], previous[2],
                        palm
                    )
                    smoothed_landmarks.append([current[0], x, y])
                current_landmarks = smoothed_landmarks

            self.previous_landmarks = current_landmarks
            landmarks = [
                [i, int(x), int(y)]
                for i, x, y in current_landmarks
            ]

        return landmarks

    def _blend(self, x, y, prev_x, prev_y, palm):
        """Follow large motion, and quiet noise that is smaller than the palm."""
        movement = math.hypot(x - prev_x, y - prev_y)
        dead = max(1.5, palm * 0.012)
        if movement <= dead:
            return prev_x, prev_y

        follow_at = max(dead + 1.0, palm * 0.10)
        opened = min((movement - dead) / (follow_at - dead), 1.0)
        alpha = self.still_mix + (1.0 - self.still_mix) * opened
        return (
            alpha * x + (1.0 - alpha) * prev_x,
            alpha * y + (1.0 - alpha) * prev_y
        )

    def fingers_up(self, lm):
        """Return [thumb, index, middle, ring, pinky]."""
        if not lm:
            return []

        def dist(a, b):
            return math.hypot(
                lm[a][1] - lm[b][1],
                lm[a][2] - lm[b][2]
            )

        palm = max(dist(0, 9), 40.0)
        margin = palm * 0.12
        previous = self.previous_fingers
        fingers = []

        thumb_out = dist(4, 17)
        thumb_in = dist(3, 17)
        if thumb_out > thumb_in + margin:
            fingers.append(1)
        elif thumb_out < thumb_in - margin:
            fingers.append(0)
        else:
            fingers.append(previous[0] if previous else int(thumb_out > thumb_in))

        for index, tip in enumerate((8, 12, 16, 20)):
            tip_y = lm[tip][2]
            pip_y = lm[tip - 2][2]
            if tip_y < pip_y - margin:
                fingers.append(1)
            elif tip_y > pip_y + margin:
                fingers.append(0)
            else:
                fallback = 1 if tip_y < pip_y else 0
                fingers.append(previous[index + 1] if previous else fallback)

        self.previous_fingers = fingers
        return fingers

    def distance(self, lm, a, b):
        """Pixel distance between two landmarks."""
        return math.hypot(
            lm[a][1] - lm[b][1],
            lm[a][2] - lm[b][2]
        )
