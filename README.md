# otb-chess-vision
A computer vision project to detect state of an over the board chess game and encode it in FEN notation.This is a simple implementation of YOLO for the chess piece detection and computer vision techniques to detect the square occupied by the pieces.

## Overview:
------------
This project approaches the problem in 3 parts: 
1. Tile Detection: 
    - For tile detection one of the most common methods, is applying canny edge to get the  edges of the image, and then applying hough transform on the image to detect all the possible straight lines for the edges. In my experience with hough transform, I have found it to be very sensitive to camera angles and resolving the duplicate lines from the transformation needed DBSCAN which does not always give the desired results.  
    - Calibrating the tiles of the empty board which are nothing but contours, and using the coordinates retrived in calibration throughout the duration of the game, seemed to be a much simpler and effective solution to the problem.
 

2. Piece Detection:
    - For the piece detection, a custom trained YOLO v8 is implemented. 
    - The YOLO v8 was trained on a custom data; the data was annotated with roboflow. 
    - The mAP50 for the model is 0.947 and mAP50-95 is 0.616. This can be further improved with more data and training.

3. Board representation:
    - Once the tiles and pieces are detected, the coordinates are used to identify which squares are occupied by the pieces. 
    - The game state is then converted to FEN notation, and the board is represented with the help of python-chess, which in future can be used for SVG rendering of the game state, provide move validation and a ton of other features.

## Demo:
![](https://github.com/anoopgeorg/otb-chess-vision/blob/8eab6e9a10f8ea1bc210fd6989a7f383f81fb5e1/demo/otb-chess-vision-demo.gif)

## TODOs:

- Better tile detection that does not need calibration and is not sensitive to occlusion by chess pieces.
- Improve piece detection model with more data and training.
- Implement a SVG server to display the board state.
