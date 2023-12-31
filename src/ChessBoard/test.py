from chessBoard import chessBoard
import cv2
from pathlib import Path


def run():
    parent = Path().absolute().parent
    img_path = (
        parent
        / "data/raw/roboflow-chess-pieces/train/images/IMG_0294_JPG.rf.cb349f708d70f8f46097636ec55f5419.jpg"
    )
    test_img_path = Path(
        "C:/Users/ANOOP/Desktop/Python/ai-porfolio/otb-chess-vision/src/ChessBoard/data"
    )
    test_images = test_img_path.glob("*.jpg")

    for images in test_images:
        img = cv2.imread(str(images))

        if img is not None:
            chessboard = chessBoard(img)
            ####board.detect_corners()
            # img,sorted_contours = chessboard.detect_tiles(chessboard.SRC_IMG)

            board, _ = chessboard.calibrateBoard(img)
            location = chessboard.boardSearch([173, 128], board)
            if location is not None:
                print(location)
            ##chessboard.detect_corners()

            # chessboard.detect_good_features()
            # chessboard.calibrate_board(img)
            # chessboard.calibrate_board()

        else:
            print(f"Error Loading image{img_path}")


if __name__ == "__main__":
    run()
