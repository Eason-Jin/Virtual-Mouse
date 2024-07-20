# https://www.youtube.com/watch?v=4NmwNEYtL1s&t=2s

import cv2
import mediapipe as mp
import pyautogui
import random
import util
from pynput.mouse import Button, Controller
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*GetPrototype.*")

mouse = Controller()
screen_width, screen_height = pyautogui.size()

mp_hands = mp.solutions.hands
handsModel = mp_hands.Hands(static_image_mode=False, model_complexity=1,
                           min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)

class Finger:
    def __init__(self, tip, mid, base):
        self.tip = tip
        self.mid = mid
        self.base = base
    
    def finger_angle(self):
        return util.get_angle(self.tip, self.mid, self.base)

def detect_gestures(frame, landmarks, results):
    if len(landmarks) >= 21:

        index_finger = Finger(landmarks[8], landmarks[7], landmarks[5])
        middle_finger = Finger(landmarks[12], landmarks[11], landmarks[9])

        thumb_tip = landmarks[4]
        if not thumb_tip:
            return

        print(f"Index angle: {round(index_finger.finger_angle(), 2)},\tMiddle angle: {round(middle_finger.finger_angle(), 2)},\tThumb distance: {round(util.get_distance(thumb_tip, index_finger.base), 2)}")
        
        # If distance less than 0.1, move mouse
        if util.get_distance(thumb_tip, index_finger.base) < 0.1:
            x = int(index_finger.tip[0] * screen_width)
            y = int(index_finger.tip[1] * screen_height)
            pyautogui.moveTo(x, y)
        # Left click
        elif index_finger.finger_angle() < 50 and middle_finger.finger_angle() > 100:
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Right click
        elif middle_finger.finger_angle() < 50 and index_finger.finger_angle() > 100:
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Double click
        elif index_finger.finger_angle() < 50 and middle_finger.finger_angle() < 50:
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def main():
    cap = cv2.VideoCapture(0)
    draw = mp.solutions.drawing_utils

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = handsModel.process(frameRGB)

            landmarks = []
            if results.multi_hand_landmarks:
                # If multiple hands just take the first one
                hand_landmarks = results.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                                    mp.solutions.drawing_styles.get_default_hand_connections_style())
                # Get the landmarks (21 per hand)
                for lm in hand_landmarks.landmark:
                    landmarks.append((lm.x, lm.y))

            detect_gestures(frame, landmarks, results)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
