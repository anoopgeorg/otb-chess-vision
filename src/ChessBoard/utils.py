import numpy as np
import cv2


# Calulates Euclidean distance between P1 and P2
def euclidieanDist(p1, p2):
    """
    Returns euclidean distance between points 1 and 2

    Args:
    - p1: Point 1 (x1,y1)
    - p2: Point 2 (x2,y2)
    """
    temp = p2 - p1
    return np.sqrt(np.dot(temp, temp.T))


# Plot the points onto an image
def draw_points(img: np.array, points: np.array):
    """
    Returns image with points displayed on the image

    Args:
    - img: Source image.
    - points: Array of points to be displayed
    """
    print(f"draw_points{points}")
    if points is not None:
        for point in points:
            x = int(point[0])
            y = int(point[1])
            plotted_img = cv2.circle(
                img, (x, y), radius=2, color=(255, 0, 0), thickness=1
            )
    return plotted_img


def medianSquares(squares):
    """
    Returns the median squares of a given board section

    Args:
    - squares: An array of squares in a board section.
    """
    length = len(squares)
    if length % 2 == 0:
        return [squares[(length // 2) - 1], squares[length // 2]]
    else:
        median = np.median(squares)
        return [median, median]
