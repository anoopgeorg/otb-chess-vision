import cv2
import numpy as np
import argparse

from chessBoard import chessBoard
from pieceDetector import pieceDetector
import utils


class chessVision:
    def __init__(self, device=0):
        """
        Initialize the chessVision class.
        """
        self.device = device
        self.pieceDetector = pieceDetector()
        self.BOARD_COORD = None

    # Calibrate the board to get board coordinates
    # Input -> Image Frame
    # Output -> Img,board_coordinates,board_object
    def calibration(self, frame):
        """
        Calibrate the chessboard.
        The A1 square has to be oriented to the right of the camera

        Args:
        - frame: input image.
        """
        print(f"FRAME -> SHAPE{frame.shape}")
        self.chessBoard = chessBoard(frame)
        # row 0 -> H file,..., 7 -> A file
        board_coord, board = self.chessBoard.calibrateBoard(frame)
        self.BOARD_COORD = board_coord
        return (board_coord, board)

    def detectState(self, frame):
        """
        Returns the current board state.

        Args:
        - frame: input image.
        Returns:
        - Board object with current state
        """
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
        """
        Open webcam stream for game state detection.

        """
        frameWidth = 640
        frameHeight = 640
        # link = "http://192.168.0.110:4747/video"
        cap = cv2.VideoCapture(self.device)
        cap.set(3, frameWidth)
        cap.set(4, frameHeight)
        cap.set(10, 150)

        if not cap.isOpened():
            print("Error opening video stream or file")

        window_msg = "Press 'Enter' to caliberate board"
        while cap.isOpened():
            success, src_img = cap.read()
            if success:
                img = cv2.resize(src_img, (640, 640))
                if self.BOARD_COORD is None:
                    # Show camera feed
                    cv2.imshow(window_msg, img)
                else:
                    # Overlay numbered tiles on camera feed
                    numbered_img = self.chessBoard.displayNumberedTiles(
                        self.BOARD_COORD, img
                    )
                    cv2.imshow(window_msg, numbered_img)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == 13:  # Check For "Enter"
                    print("Claibration started:")
                    (board_coords, board) = self.calibration(img)
                    if board_coords is not None:
                        window_msg = (
                            "Press 'p' for prediction,'Enter' for recalibration "
                        )

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
    parser = argparse.ArgumentParser(description="Get FEN encoding of OTB chess game")
    parser.add_argument("--device", metavar="device", type=int, help="Webcam id")
    parser.add_argument("--link", metavar="link", type=str, help="Stream url")

    args = parser.parse_args()
    device = args.device if args.device is not None else args.link
    device = device if device is not None else 0

    test = chessVision(device=device)
    try:
        test.streamCapture()
    except Exception as e:
        print(e)
    # test.test()
