# RULES (https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker):
# Thumb out: mouse move, otherwise mouse pause
#   - Solution: find distance between landmark 4 and 5
# Index bend: mouse left click
#   - Solution: find angle between 5 -> 6 and 6 -> 8 while middle finger is straight
# Middle bend: mouse right click
#   - Solution: find angle between 9 -> 10 and 10 -> 12 while index finger is straight
# Both bend: mouse double click
#   - Solution: find angle between 5 -> 6 and 9 -> 10

import numpy as np

def get_angle(p1, p2, p3):
    # Angle of A(ax, ay) and B(bx, by) with the x axis is given by: arctan((by-ay)/(bx-ax))
    rad = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    return np.abs(np.degrees(rad))

def get_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)