import cv2
import numpy as np
import utils
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from ultralytics import YOLO
import chess
from scipy import stats as st


class chessBoard:
    def __init__(self, src_img, weights_path: str = None):
        self.SRC_IMG = cv2.resize(src_img, (640, 640))
        self.H = self.SRC_IMG.shape[0]
        self.W = self.SRC_IMG.shape[1]
        self.GRAY_IMG = cv2.cvtColor(self.SRC_IMG, cv2.COLOR_BGR2GRAY)

    def show_board(self, title, img):
        # cv2.imshow(title, img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        pass

    def find_anchor_points(self, xys):
        anchor = np.zeros((2, 2, 2))
        temp = xys.numpy().copy()
        x_sorted = temp[:, 0].argsort()

        # Get the Left and Right side anchor points
        left_anchors = temp[x_sorted[:2]]
        right_anchors = temp[x_sorted[2:]]

        # Determine the top and bottom corners
        top_left = left_anchors[left_anchors[:, 1].argmax()]
        bottom_left = left_anchors[left_anchors[:, 1].argmin()]

        top_right = right_anchors[right_anchors[:, 1].argmax()]
        bottom_right = right_anchors[right_anchors[:, 1].argmin()]

        #### print(top_left)
        #### print(top_right)
        #### print(bottom_left)
        #### print(bottom_right)

        # Fill the anchor with the corners based on thier locations
        anchor[0, 0], anchor[1, 0], anchor[0, 1], anchor[1, 1] = (
            top_left,
            bottom_left,
            top_right,
            bottom_right,
        )
        return anchor

    def auto_canny(self, img, sigma=0.35):
        ####print("Edge extraction started")
        median = np.median(img)
        lower_bound = int(max(0, (1.0 - sigma) * median))
        upper_bound = int(max(0, (1.0 + sigma) * median))
        img_edges = cv2.Canny(img, lower_bound, upper_bound)
        self.show_board("Canny Edge", img_edges)
        return img_edges

    def draw_points(self, img: np.array, points: np.array):
        ####print(f"draw_points{points}")
        if points is not None:
            for point in points:
                x = point[0]
                y = point[1]
                plotted_img = cv2.circle(
                    img, (x, y), radius=2, color=(255, 0, 0), thickness=1
                )

            self.show_board("poi_img", plotted_img)

    def find_clusters(self, points):
        dbscan = DBSCAN(eps=40, min_samples=3)
        model = dbscan.fit(points)
        labels = model.labels_
        ####print(f"<+++++++++++++++++>")
        ####print(f"Clusters Found{len(labels)}")
        ####print(labels)

    # Get the shapes in a given threshold image
    # Input -> Threshold image
    # Output-> list of  [x1,y1,x2,y2,area]
    def find_shapes(self, threshold_mat):
        contours, hierarchy = cv2.findContours(
            threshold_mat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        ###
        test = threshold_mat.copy()
        cv2.drawContours(test, contours, -1, (255, 0, 0))
        self.show_board("ALL CONTOURS DRAWN", test)
        ###

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

        ####print(f"contours{len(contours)}")
        # Get the co-ordinates and areas of each contour
        contour_properties = []
        i = 0
        for contour in contours:
            if i in valid_indices:
                con_areas = cv2.contourArea(contour)
                ####print(f"Area{con_areas}")
                x, y, w, h = cv2.boundingRect(contour)
                # # Check for squarish shapes
                if (h != 0) and (min(w, h) / max(w, h) >= 0.55):
                    row = [x, y, (x + w), (y + h), w, h, con_areas]
                    contour_properties.append(row)
            i = i + 1
        contour_properties = np.array(contour_properties).astype("intc")
        ###
        for contour in contour_properties:
            ####print(contour)
            cv2.rectangle(
                threshold_mat,
                (int(contour[0]), int(contour[1])),
                (int(contour[2]), int(contour[3])),
                (255, 0, 0),
                thickness=1,
            )
        self.show_board("all contours", threshold_mat)
        ###

        return np.array(contour_properties).astype("intc")

    # Returns a mask based on contours
    def getContourMask(self, contour_properties):
        # Generate a contour mask for getting avg pixel values in the contour region
        mask = np.zeros_like(self.GRAY_IMG)
        for contour in contour_properties:
            ####print(contour)
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
        ####print(f"median ||||| {median}")
        indices = np.where((areas >= (0.15 * median)) & (areas <= (3 * median)))
        ret_contours = contour_properties[indices]
        ###
        for contour in ret_contours:
            ####print(contour)
            cv2.rectangle(
                img,
                (int(contour[0]), int(contour[1])),
                (int(contour[2]), int(contour[3])),
                (255, 0, 0),
                thickness=1,
            )
        self.show_board("after area resolve contours", img)
        ###

        return ret_contours

    def draw_tiles(self, contours, img, n_contours=64):
        contour_properties = self.resolve_contours(contours, img)  ### img passed
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
        ####print(f"CENTROID ==>{centroid}")

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

        # find 32 contours(Dark tiles) that are the nearest to the centroid
        # sorted_points = points[points[:,-1].argsort()][:n_contours]
        sorted_points = points[dist_to_centroid.argsort()][:n_contours]

        # Extract the location of light tiles based on the location of dark tiles
        contour_mask = self.getContourMask(sorted_points)
        self.show_board("Contour Mask sorted", contour_mask)
        # Image morphings for easier recognition of basic image features
        element = cv2.getStructuringElement(1, (5, 5), (3, 3))
        contour_mask = cv2.dilate(contour_mask, element)
        self.show_board("dilated_contour_mask", contour_mask)
        contour_mask = cv2.blur(contour_mask, (9, 9), 0)
        self.show_board("blured_dilated_contour_mask", contour_mask)

        ####print(f"====>>len(sorted_points){len(sorted_points)}")
        ####print(f"====>>sorted_points.shape{sorted_points.shape}")

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

        # image = cv2.dilate(image, (9, 9), 0)
        # self.show_board("NA - Dilated edges", image)

        image = cv2.GaussianBlur(image, (11, 11), 0)
        self.show_board("NA - Blurred Dilated edges", image)

        # Find the contours
        thresh_contours = self.find_shapes(image)
        return_img = self.SRC_IMG.copy()
        return_img, sorted_contours = self.draw_tiles(thresh_contours, return_img)
        self.show_board("final_detection", return_img)
        return return_img, sorted_contours

    def pruneSaddle(self, s):
        thresh = 128
        score = (s > 0).sum()
        while score > 10000:
            ####print(thresh)
            thresh = thresh * 2
            s[s < thresh] = 0
            score = (s > 0).sum()

    def getSaddle(self, gray_img):
        img = gray_img.astype(np.float64)
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
        gxx = cv2.Sobel(gx, cv2.CV_64F, 1, 0)
        gyy = cv2.Sobel(gy, cv2.CV_64F, 0, 1)
        gxy = cv2.Sobel(gx, cv2.CV_64F, 0, 1)

        S = gxx * gyy - gxy**2
        return S

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
                ####print(f"Before sorted_coords[i:i+8]{sorted_coords[i:i+8]}")
                sorted_coords[i : i + 8] = sorted(
                    sorted_coords[i : i + 8], key=lambda x: (x[0])
                )
                ####print(f"After sorted_coords[i:i+8]{sorted_coords[i:i+8]}")

            # board coordinates
            # row 0 -> H file,..., 7 -> A file
            board = np.array(sorted_coords, dtype="int").reshape(8, 8, 2)
            img_numbered = self.displayNumberedTiles(board)
            self.show_board("numbered tiles", img_numbered)
            ####print(board)
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
                    ####print(f"i = {i} , cord= {tile}")
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
        ####print(centroid)

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
        # base_fen = str(("." * 8 + "/") * 8).rstrip("/")
        # piece_list = pieces[:, 0]
        # row_inx = pieces[:, 1].astype("int") * 8
        # col_inx = pieces[:, 2].astype("int")
        # square_inx = row_inx + col_inx  # row_index * 8 + column_index
        # new_fen = row_string = ""
        # square_count = 1
        # for i, c in enumerate(base_fen):
        #     count_reset = False
        #     # Check if piece occupies a square
        #     if np.isin(square_inx, i):
        #         piece_inx = np.where(square_inx == i)
        #         chess_piece = piece_list[piece_inx[0][0]]
        #         c = chess_piece
        #         count_reset = True
        #         count = str(square_count) if square_count != 0 else ""
        #         row_string = row_string + count + c

        #     # Check if end of row
        #     if c == "/":
        #         new_fen = new_fen + row_string + "/"
        #         count_reset = True
        #         row_string = ""

        #     row_string = str(square_count)
        #     if count_reset:
        #         square_count = 1
        #     else:
        #         square_count = square_count + 1

        # print(new_fen)
