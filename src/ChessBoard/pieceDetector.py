from ultralytics import YOLO
from pathlib import Path
import numpy as np


class pieceDetector:
    def __init__(self):
        """
        Initialize the pieceDetector object and detection model.
        """
        self.PATH = Path("..") / "src/models/chess-piece-detector.pt"
        self.FEN_PIECE = {
            "black-bishop": "b",
            "black-king": "k",
            "black-knight": "n",
            "black-pawn": "p",
            "black-queen": "q",
            "black-rook": "r",
            "white-bishop": "B",
            "white-king": "K",
            "white-knight": "N",
            "white-pawn": "P",
            "white-queen": "Q",
            "white-rook": "R",
        }
        self.model = self.loadModel(self.PATH)
        if self.model is not None:
            self.class_names = self.model.names
        else:
            self.class_names = None

    def loadModel(self, path):
        """
        Load the piece detection model during initialization.

        Args:
        - path: model path.
        Returns:
        - YOLO model
        """
        model = YOLO(path)
        model.fuse()
        return model

    # Get the predictions on an image
    def getPrediction(self, frame):
        """
        Get the prediction of pieces in an image

        Args:
        - frame: Source image.
        Returns:
        - Results object
        """
        results = self.model.predict(frame)
        return results

    # Get the coordinates of the pieces
    def getPieceCoordinates(self, results):
        """
        Get the coordinates of the pieces

        Args:
        - results: Results object.
        Returns:
        - list of piece coordinates
        """
        coordinates = []
        for result in results:
            xywhs = result.boxes.xywh.numpy()
            cls = result.boxes.cls.numpy()
            # Contains the class first and then xywh
            box = np.c_[cls, xywhs]
            coordinates.append(box)
        return coordinates

    # Get the classification name
    def getClassName(self, cls):
        """
        Get the class name from class dictionary

        Args:
        - cls: class detected by model.
        Returns:
        - Name of the class
        """
        if self.class_names is not None:
            try:
                name = self.class_names[cls]
                return name
            except Exception as e:
                print(e)
        else:
            return None

    # Get the FEN for a piece
    def getFenPiece(self, cls):
        """
        Get the FEN symbol for detected class

        Args:
        - cls: class detected by model.
        Returns:
        - FEN symbol for the piece
        """
        name = self.getClassName(cls)
        if name is not None:
            fen = self.FEN_PIECE[name]
            return fen
        else:
            return None

    # Get the lowered center
    def getLoweredCenter(self, coordinates_list):
        """
        Get the lowered center of piece for more accurate
        detection of occupied squares.

        Args:
        - coordinates_list: List of piece coordinates.
        Returns:
        - List of adjusted coordinates
        """
        new_coordinates_list = []
        for coordinates in coordinates_list:
            # 0 -> class; (1,2) -> x,y ; (3,4) -> w,h
            # new y = old y + (h/4)
            coordinates[:, 2] = coordinates[:, 2] + (coordinates[:, 4] // 4)
            new_coordinates_list.append(coordinates)
        return new_coordinates_list

    # Get the piece  locations
    def getPieceLocation(self, frame):
        """
        Get piece location coordinates in the image.

        Args:
        - frame: Input image.
        Returns:
        - List of piece coordinates
        """
        results = self.getPrediction(frame)
        if results is not None:
            coordinates = self.getPieceCoordinates(results)
            coordinates = self.getLoweredCenter(coordinates)

            return np.array(coordinates)
        else:
            print("No piece detected!")
            return None
