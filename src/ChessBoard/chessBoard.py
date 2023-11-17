import cv2
import numpy as np
import chess

import utils


class chessBoard:
    def __init__(self, src_img):
        self.SRC_IMG = cv2.resize(src_img, (640, 640))
        self.H = self.SRC_IMG.shape[0]
        self.W = self.SRC_IMG.shape[1]
        self.GRAY_IMG = cv2.cvtColor(self.SRC_IMG, cv2.COLOR_BGR2GRAY)

    def show_board(self, title, img, debug=False):
        if debug == True:
            cv2.imshow(title, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def auto_canny(self, img, sigma=0.35):
        median = np.median(img)
        lower_bound = int(max(0, (1.0 - sigma) * median))
        upper_bound = int(max(0, (1.0 + sigma) * median))
        img_edges = cv2.Canny(img, lower_bound, upper_bound)
        self.show_board("Canny Edge", img_edges)
        return img_edges

    def draw_points(self, img: np.array, points: np.array):
        if points is not None:
            for point in points:
                x = point[0]
                y = point[1]
                plotted_img = cv2.circle(
                    img, (x, y), radius=2, color=(255, 0, 0), thickness=1
                )

            self.show_board("poi_img", plotted_img)

    # Get the shapes in a given threshold image
    # Input -> Threshold image
    # Output-> list of  [x1,y1,x2,y2,area]
    def find_shapes(self, threshold_mat):
        contours, hierarchy = cv2.findContours(
            threshold_mat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # Resolve and accept only contours that have similar parents
        # Only the parents with the most children are accepted
        hierarchy = hierarchy.squeeze()
        parents = hierarchy[:, -1]
        parents_mode, counts = np.unique(parents, return_counts=True)
        max_counts = counts.argsort()[::-1]
        parent_ids = []
        count_sum = 0
        for count_idx in max_counts:
            parent_ids.append(count_idx)
            count_sum = counts[count_idx] + count_sum
            if count_sum >= 64:
                break
        valid_indices = np.where(np.isin(parents, parents_mode[parent_ids]))[0]

        # Get the co-ordinates and areas of each contour
        contour_properties = []
        i = 0
        for contour in contours:
            if i in valid_indices:
                con_areas = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                # # Check for squarish shapes
                if (h != 0) and (min(w, h) / max(w, h) >= 0.55):
                    row = [x, y, (x + w), (y + h), w, h, con_areas]
                    contour_properties.append(row)
            i = i + 1
        contour_properties = np.array(contour_properties).astype("intc")

        return np.array(contour_properties).astype("intc")

    # Returns a mask based on contours
    def getContourMask(self, contour_properties):
        # Generate a contour mask for getting avg pixel values in the contour region
        mask = np.zeros_like(self.GRAY_IMG)
        for contour in contour_properties:
            cv2.rectangle(
                mask,
                (int(contour[0][0]), int(contour[0][1])),
                (int(contour[2][0]), int(contour[2][1])),
                (255, 255, 255),
                thickness=-1,
            )

        return mask

    # Resolve the contour list to get only the required contours
    # Input -> List of [x1,y1,x2,y2,x3,y3,x4,y4,w,h,area]
    # Output-> list of [x1,y1,x2,y2,x3,y3,x4,y4,w,h,area]
    def resolve_contours(
        self, contour_properties, img=np.full((640, 640, 3), (0, 0, 0))
    ):
        # find lower and upper range values
        areas = contour_properties[:, -1]
        median = np.median(areas)

        indices = np.where((areas >= (0.15 * median)) & (areas <= (3 * median)))
        ret_contours = contour_properties[indices]
        ###BOC -Useful to debug issues in contour resolution
        # for contour in ret_contours:
        #     ####print(contour)
        #     cv2.rectangle(
        #         img,
        #         (int(contour[0]), int(contour[1])),
        #         (int(contour[2]), int(contour[3])),
        #         (255, 0, 0),
        #         thickness=1,
        #     )
        # self.show_board("after area resolve contours", img)
        ###EOC -Useful to debug issues in contour resolution

        return ret_contours

    def draw_tiles(self, contours, img, n_contours=64):
        contour_properties = self.resolve_contours(contours, img)
        # Centroid of all the contour points should ideally be the middle of the chessboard
        t_left = np.array([contour_properties[:, 0], contour_properties[:, 1]])  # x1,y1
        t_right = np.array(
            [
                contour_properties[:, 0] + contour_properties[:, 4],
                contour_properties[:, 1],
            ]
        )  # x1+w,y1
        b_right = np.array(
            [contour_properties[:, 2], contour_properties[:, 3]]
        )  # x2,y2
        b_left = np.array(
            [
                contour_properties[:, 2] - contour_properties[:, 4],
                contour_properties[:, 3],
            ]
        )  # x2-w,y2
        t_left_m = t_left.mean(axis=1)
        t_right_m = t_right.mean(axis=1)
        b_right_m = b_right.mean(axis=1)
        b_left_m = b_left.mean(axis=1)
        centroid = np.mean([t_left_m, t_right_m, b_right_m, b_left_m], axis=0).astype(
            "int"
        )

        # Gather the dimensions to reduce to the required contours
        points = []
        dist_to_centroid = []
        for con in contour_properties:
            # cv2.rectangle(img,(con[0],con[1]),(con[2],con[3]),(255,0,0),2)
            x1, y1 = con[0], con[1]  # Top Left (x1,y1 )
            x2, y2 = con[0] + con[4], con[1]  # Top Right (x1+w,y1)
            x3, y3 = con[2], con[3]  # Bottom Right (x2,y2)
            x4, y4 = con[2] - con[4], con[3]  # Bottom Left (x2-w,y2 )
            distane_to_centroid = utils.euclidieanDist(np.array([x1, y1]), centroid)

            points.append([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
            dist_to_centroid.append(distane_to_centroid)

        points = np.array(points)
        dist_to_centroid = np.array(dist_to_centroid)

        # find 64 contours(all tiles) that are the nearest to the centroid
        sorted_points = points[dist_to_centroid.argsort()][:n_contours]

        # Extract the location all tiles
        contour_mask = self.getContourMask(sorted_points)
        self.show_board("Contour Mask sorted", contour_mask)
        # Image morphings for easier recognition of basic image features
        element = cv2.getStructuringElement(1, (5, 5), (3, 3))
        contour_mask = cv2.dilate(contour_mask, element)
        self.show_board("dilated_contour_mask", contour_mask)
        contour_mask = cv2.blur(contour_mask, (9, 9), 0)
        self.show_board("blured_dilated_contour_mask", contour_mask)

        for point in sorted_points:
            cv2.rectangle(
                img,
                (int(point[0][0]), int(point[0][1])),
                (int(point[2][0]), int(point[2][1])),
                (255, 0, 0),
                -1,
            )
        aplha = 0.33
        ret_img = cv2.addWeighted(img, aplha, self.SRC_IMG.copy(), 1 - aplha, 0)
        return ret_img, sorted_points

    def detect_tiles(self, img):
        image = self.GRAY_IMG.copy()
        self.show_board("NA - Image OG", image)

        image = cv2.GaussianBlur(image, (7, 7), 0)
        self.show_board("NA - Gaussian Blur", image)

        image = self.auto_canny(image)
        self.show_board("NA - Canny", image)

        image = cv2.GaussianBlur(image, (11, 11), 0)
        self.show_board("NA - Blurred Dilated edges", image)

        # Find the contours
        thresh_contours = self.find_shapes(image)
        return_img = self.SRC_IMG.copy()
        return_img, sorted_contours = self.draw_tiles(thresh_contours, return_img)
        self.show_board("final_detection", return_img)
        return return_img, sorted_contours

    # Calibrates the empty board before the game starts
    # Calibration is done with A1 to the right of the camera
    # Coordinates are saved for refrence throughout the game
    def calibrateBoard(self, img):
        # Get the contours of the board and detect the tiles
        img, contours = self.detect_tiles(img)
        if contours is not None and len(contours) == 64:
            print("Board Found!")
            # Find the centroids of the tiles and sort based on Y-axis
            tile_centroids = contours.mean(axis=1).astype("int")
            sorted_coords = sorted(tile_centroids, key=lambda x: (x[1]))
            board = np.zeros((8, 8, 2), dtype="int")
            # Sort each row of 8 on X-axis
            for i in np.arange(0, 72, 8):
                sorted_coords[i : i + 8] = sorted(
                    sorted_coords[i : i + 8], key=lambda x: (x[0])
                )
            # board coordinates
            # row 0 -> H file,..., 7 -> A file
            board = np.array(sorted_coords, dtype="int").reshape(8, 8, 2)
            img_numbered = self.displayNumberedTiles(board)
            self.show_board("numbered tiles", img_numbered)
            return board, chess.Board().clear_board()

        else:
            print(f"{len(contours)} tiles detected")
            print("Please adjust the camera angle and recalibrate the board")
            return None, None

    # Displays numbered tiles for given coordinates
    def displayNumberedTiles(self, board):
        img_c = self.SRC_IMG.copy()
        # Go through each file
        if board is not None:
            for f in board:
                # Print each tile in the file
                for i, tile in enumerate(f):
                    cv2.putText(
                        img_c,
                        str(i),
                        tile,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.25,
                        (255, 255, 255),
                        1,
                    )
        return img_c

    # Search the board for the tile closest to the piece center
    def boardSearch(self, piece_center, board_coords, file_win=[0, 7], rank_win=[0, 7]):
        testing = self.SRC_IMG.copy()
        if (file_win[0] == file_win[1]) and (rank_win[0] == rank_win[1]):
            return [file_win[0], rank_win[1]]

        # Select the sectiion of the board
        file_range = np.arange(file_win[0], file_win[1] + 1)
        file_median = utils.medianSquares(file_range)

        rank_range = np.arange(rank_win[0], rank_win[1] + 1)
        rank_median = utils.medianSquares(rank_range)

        center_section = board_coords[
            file_median[0] : file_median[1] + 1, rank_median[0] : rank_median[1] + 1
        ]
        centroid = np.mean(center_section.mean(axis=0), axis=0)

        board_section = board_coords[
            file_win[0] : file_win[1] + 1, rank_win[0] : rank_win[1] + 1
        ]

        # Right half
        if piece_center[0] > centroid[0]:
            # new_file_win = [file_median[1], file_range[-1]]
            new_rank_win = [rank_median[1], rank_range[-1]]

            # Right bottom quadrant
            if piece_center[1] > centroid[1]:
                # new_rank_win = [rank_median[1], rank_range[-1]]
                new_file_win = [file_median[1], file_range[-1]]
            # Right top quadrant
            elif piece_center[1] < centroid[1]:
                # new_rank_win = [rank_range[0], rank_median[0]]
                new_file_win = [file_range[0], file_median[0]]
        # Left half
        elif piece_center[0] < centroid[0]:
            # new_file_win = [file_range[0], file_median[0]]
            new_rank_win = [rank_range[0], rank_median[0]]
            # Left bottom quadrant
            if piece_center[1] > centroid[1]:
                # new_rank_win = [rank_median[1], rank_range[-1]]
                new_file_win = [file_median[1], file_range[-1]]
            # Left top quadrant
            elif piece_center[1] < centroid[1]:
                # new_rank_win = [rank_range[0], rank_median[0]]
                new_file_win = [file_range[0], file_median[0]]
        else:
            print("equal to centroid")

        # Plot the center
        cv2.circle(testing, (int(centroid[0]), int(centroid[1])), 2, (0, 0, 255), 1)
        cv2.circle(
            testing, (int(piece_center[0]), int(piece_center[1])), 4, (255, 255, 255), 1
        )
        # Plot the square centers in given section

        for row in center_section:
            for square in row:
                cv2.circle(testing, (int(square[0]), int(square[1])), 2, (255, 0, 0), 1)

        self.show_board("Centeroid and center section", testing)

        return self.boardSearch(piece_center, board_coords, new_file_win, new_rank_win)

    def getBoardRepresentation(self, pieces):
        # Get an empty board
        board = np.full((8, 8), ".")

        # Place pieces on the board
        for piece_details in pieces:
            piece = piece_details[0]
            row = int(piece_details[1])
            col = int(piece_details[2])
            board[row, col] = piece

        ####print(board)
        board = np.rot90(board, -1)  # Reorient the board before display
        fen = ""
        for row in board:
            square_count = 0
            r = ""
            for square_content in row:
                if square_content == ".":
                    square_count = square_count + 1
                else:
                    if square_count == 0:
                        r = r + square_content
                    else:
                        r = r + str(square_count) + square_content
                    square_count = 0
            if len(r) == 0:
                fen = fen + str(square_count) + "/"
            else:
                if square_count == 0:
                    fen = fen + r + "/"
                else:
                    fen = fen + r + str(square_count) + "/"
        fen = fen.rstrip("/")
        board = chess.Board(fen)
        print(board)
