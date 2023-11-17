import cv2
import numpy as np

from chessBoard import chessBoard
from pieceDetector import pieceDetector
import utils


class chessVision:
    def __init__(self):
        self.pieceDetector = pieceDetector()
        self.BOARD_COORD = None

    # Caliberate the board to get board coordinates
    # Input -> Image Frame
    # Output -> Img,board_coordinates,board_object
    def calibration(self, frame):
        print(f"FRAME -> SHAPE{frame.shape}")
        self.chessBoard = chessBoard(frame)
        # row 0 -> H file,..., 7 -> A file
        board_coord, board = self.chessBoard.calibrateBoard(frame)
        img = self.chessBoard.displayNumberedTiles(board_coord)
        self.BOARD_COORD = board_coord
        return (img, board_coord, board)

    # Detect the state of the board
    # Input -> Image Frame
    # Output -> Board object
    def detectState(self, frame):
        if self.BOARD_COORD is not None and self.pieceDetector is not None:
            results = self.pieceDetector.getPieceLocation(frame)
            for result in results:
                centroids = result[:, 1:3]
                img = utils.draw_points(img=frame, points=centroids)

            # Find the square for each piece
            pieces = []
            for result in results:
                for coordinate in result:
                    piece = self.pieceDetector.getFenPiece(coordinate[0])
                    piece_center = coordinate[1:3]
                    square = self.chessBoard.boardSearch(piece_center, self.BOARD_COORD)
                    if square is not None:
                        pieces.append([piece, square[0], square[1]])

            # Get the board representation
            pieces = np.array(pieces)
            self.chessBoard.getBoardRepresentation(pieces)
            return img

    def streamCapture(self):
        frameWidth = 640
        frameHeight = 640
        link = "http://192.168.0.110:4747/video"
        cap = cv2.VideoCapture(link)
        cap.set(3, frameWidth)
        cap.set(4, frameHeight)
        cap.set(10, 150)

        if not cap.isOpened():
            print("Error opening video stream or file")

        window_msg = "Press Space once the black tiles are recognized"
        while cap.isOpened():
            sucess, src_img = cap.read()
            if sucess:
                img = cv2.resize(src_img, (640, 640))
                if self.BOARD_COORD is None:
                    (tiles_img, _, _) = self.calibration(img)
                    cv2.imshow(window_msg, tiles_img)
                else:
                    cv2.imshow(window_msg, img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == 13:  # Check For "Enter"
                    print("Claibration started:")
                    (tiles_img, board_coord, board) = self.calibration(img)
                    if tiles_img is not None:
                        window_msg = (
                            "Now set the pieces, then press 'p' to get prediction"
                        )
                        cv2.imshow("Detected tiles, press enter to confirm", tiles_img)

                if key == ord("p"):
                    points_img = self.detectState(img)
                    if points_img is not None:
                        cv2.imshow("Detected pieces", points_img)

            else:
                print("Some issue with camera")
                break

        cap.release()
        cv2.destroyAllWindows()

    def test(self):
        img_path = "C:/Users/ANOOP/Desktop/Python/ai-porfolio/otb-chess-vision/src/ChessBoard/img.jpg"
        img2_path = "C:/Users/ANOOP/Desktop/Python/ai-porfolio/otb-chess-vision/src/ChessBoard/img2.jpg"

        img = cv2.resize(cv2.imread(img_path), (640, 640))
        img2 = cv2.resize(cv2.imread(img2_path), (640, 640))
        (tiles_img, _, _) = self.calibration(img)
        points_img = self.detectState(img2)


if __name__ == "__main__":
    test = chessVision()
    test.streamCapture()
    # test.test()
