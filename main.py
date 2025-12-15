import cv2
import numpy as np
from typing import Final

# increase filtering to remove more noise (over 10 may lead to over-smoothing)
FILTERING: Final[int] = 5
cap = cv2.VideoCapture(0)

# I know the instructions said just for blue, but this makes it work for multiple colors
# dictionary of colors including name, color range (red wraps around, so there are two ranges), and color of the box
# that identifies it
colors = [
    {
        "name": "Blue",
        "ranges": [(np.array([100, 50, 50]), np.array([130, 255, 255]))],
        "box_color": (255, 0, 0)
    },
    {
        "name": "Red",
        "ranges": [
            (np.array([0, 150, 50]), np.array([10, 255, 255])),
            (np.array([170, 150, 50]), np.array([179, 255, 255]))
        ],
        "box_color": (0, 0, 255)
    },
    {
        "name": "Green",
        "ranges": [(np.array([35, 70, 50]), np.array([85, 255, 255]))],
        "box_color": (0, 255, 0)}
]

kernel = np.ones((FILTERING, FILTERING), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # iterating through the colors
    for color in colors:

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for lower, upper in color["ranges"]:
            mask_range = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_or(mask, mask_range)

        # cleans noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        total_color_pixels = cv2.countNonZero(mask)
        if total_color_pixels == 0:
            continue

        # detect all areas of a certain color
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # largest color
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        roi_mask = mask[y:y + h, x:x + w]
        largest_color_pixels = cv2.countNonZero(roi_mask)

        percent_of_color = (largest_color_pixels / total_color_pixels) * 100

        # draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), color["box_color"], 2)
        cv2.putText(frame,
                    f'{color["name"]}: {percent_of_color:.1f}%',
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color["box_color"],
                    2)

    cv2.imshow("Multi-Color Tracking", frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()