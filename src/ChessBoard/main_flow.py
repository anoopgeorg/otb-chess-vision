from Board import Board
import cv2
from pathlib import Path


def run():      
    parent   = Path().absolute().parent 
    img_path  = parent / "data/raw/roboflow-chess-pieces/train/images/IMG_0294_JPG.rf.cb349f708d70f8f46097636ec55f5419.jpg"
    test_img_path = Path("C:/Users/ANOOP/Desktop/Python/ai-porfolio/otb-chess-vision/src/ChessBoard/img_tests")
    test_images = test_img_path.glob('*.jpg')
    for images in test_images: 
        img = cv2.imread(str(images))
        
        if img is not None:
            board = Board(src_img=img)            
            ####board.detect_corners()
            board.detect_tiles()
            
        else:
            print(f"Error Loading image{img_path}")
        

if __name__ == "__main__":
    run()