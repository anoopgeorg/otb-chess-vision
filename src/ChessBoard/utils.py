import numpy as np
import cv2


# Calulates Euclidean distance between P1 and P2
def euclidieanDist(p1, p2):
    temp = p2 - p1
    return np.sqrt(np.dot(temp, temp.T))


# Plot the points onto an image
def draw_points(img: np.array, points: np.array):
    print(f"draw_points{points}")
    if points is not None:
        for point in points:
            x = int(point[0])
            y = int(point[1])
            plotted_img = cv2.circle(
                img, (x, y), radius=2, color=(255, 0, 0), thickness=1
            )
    return plotted_img


def medianSquares(array):
    length = len(array)
    if length % 2 == 0:
        return [array[(length // 2) - 1], array[length // 2]]
    else:
        median = np.median(array)
        return [median, median]
