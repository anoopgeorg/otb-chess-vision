import cv2
import chessBoard

frameWidth = 640
frameHeight = 480

link = "http://192.168.0.110:4747/mjpegfeed?640x480"
cap = cv2.VideoCapture(link)

if not cap.isOpened():
    print("Error opening video stream or file")
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

while cap.isOpened():
    sucess, img = cap.read()
    if sucess:
        chessboard = chessBoard(src_img=img)
        # img = thresh_img = cv2.adaptiveThreshold(chessboard.GRAY_IMG,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,101,25)
        cv2.imshow("Press Space once the black tiles are recognized", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == 13:  # Check For "Enter"
            print(
                "Claibration started=================================================================== >>>>>>>>>>>>>>>>>>"
            )

            cliberation_img, sorted_contours = chessboard.calibrateBoard(img)
            ## TODO
            # ->Use the calibrated img to find the diagonal of the board
            # ->Use the diagonal to extrpolate the remaining two corner
            # ->Use the extrpolated corners to transform perspective
            # ->Once perspective is transformed, use binary mask to again detect black tiles
            # ->Use the inverse of the binary mask to detect the white tiles
            # ->Finalize the grid and arange the grid in arrays with the board rank names

            cv2.imshow("Detected Tiles", cliberation_img)
    else:
        print("Some issue with camera")
        break

cap.release()
cv2.destroyAllWindows()
