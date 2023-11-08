import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import pandas as pd


class Chessboard:
    def __init__(self, src_img, weights_path: str = None):
        self.SRC_IMG = cv2.resize(src_img, (416, 416))
        self.H = self.SRC_IMG.shape[0]
        self.W = self.SRC_IMG.shape[1]
        self.GRAY_IMG = cv2.cvtColor(self.SRC_IMG, cv2.COLOR_BGR2GRAY)
        self.BOARD = None

    def show_board(self, title, img):
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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

        print(top_left)
        print(top_right)
        print(bottom_left)
        print(bottom_right)
        # Fill the anchor with the corners based on thier locations
        anchor[0, 0], anchor[1, 0], anchor[0, 1], anchor[1, 1] = (
            top_left,
            bottom_left,
            top_right,
            bottom_right,
        )
        return anchor

    def auto_canny(self, img, sigma=0.35):
        print("Edge extraction started")
        median = np.median(img)
        lower_bound = int(max(0, (1.0 - sigma) * median))
        upper_bound = int(max(0, (1.0 + sigma) * median))
        img_edges = cv2.Canny(img, lower_bound, upper_bound)
        self.show_board("Canny Edge", img_edges)
        return img_edges

    def smoothen_image(self, img):
        img_c = img.copy()
        img_c = cv2.blur(img_c, (9, 9), 0)
        self.show_board("smooothen_blur", img_c)
        print(type(img_c))

        element = cv2.getStructuringElement(1, (7, 7), (3, 3))
        img_c = cv2.dilate(img_c, element)
        self.show_board("dialated_blur", img_c)

        img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)

        return img_c

    def draw_points(self, img: np.array, points: np.array):
        print(f"draw_points{points}")
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
        print(f"<+++++++++++++++++>")
        print(f"Clusters Found{len(labels)}")
        print(labels)

    def detect_good_features(self):
        manipulated_img = self.GRAY_IMG
        element = cv2.getStructuringElement(1, (9, 9), (3, 3))
        manipulated_img = cv2.dilate(manipulated_img, element)
        self.show_board("dilate", manipulated_img)
        manipulated_img = cv2.cvtColor(manipulated_img, cv2.COLOR_BGR2GRAY)
        self.show_board("gray_img", manipulated_img)

        corners = cv2.goodFeaturesToTrack(
            manipulated_img,
            maxCorners=300,
            qualityLevel=0.2,
            minDistance=25,
            useHarrisDetector=True,
            k=0.1,
        )
        corners = np.uint8(corners)
        img_c = self.SRC_IMG.copy()
        for c in corners:
            x, y = c.ravel()
            img_c = cv2.circle(img_c, (x, y), 3, (255, 0, 0), -1)
        self.show_board("goodfeatures", img_c)

    def detect_corners(self):
        self.show_board("gray_image", self.GRAY_IMG)

        manipulated_img = self.smoothen_image(self.SRC_IMG)
        self.show_board("manipulated_img", manipulated_img)

        gray = np.float32(manipulated_img)
        # Detector parameters
        blockSize = 2
        apertureSize = 3
        k = 0.04

        dst = cv2.dilate(cv2.cornerHarris(gray, blockSize, apertureSize, k), None)
        self.show_board("dst_img", dst)
        ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
        dst = np.uint8(dst)

        # # find centroids for all the points found by harris corner
        # ret,labels,sats,centroids = cv2.connectedComponentsWithStats(dst)
        # print(type(ret),type(labels),type(sats),type(centroids))
        # print(ret,labels.shape,sats.shape,centroids.shape)

        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        # corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
        # # Now draw them
        # res = np.hstack((centroids,corners))
        # res = np.int0(res)
        # corner_output = self.SRC_IMG.copy()
        # corner_output[res[:,1],res[:,0]]=[0,0,255]
        # corner_output[res[:,3],res[:,2]] = [0,255,0]
        # self.show_board("final_corners",corner_output)

        corner_output = self.SRC_IMG.copy()
        criteria = dst > 0.05 * dst.max()
        corner_output[criteria] = [0, 255, 0]

        # poi = np.array(np.where(criteria)).T
        idx = np.where(criteria)
        # poi = self.find_clusters(np.array(list(zip(idx[1], idx[0]))))
        poi = np.array(list(zip(idx[1], idx[0])))
        self.draw_points(self.SRC_IMG, poi)
        self.show_board("final_corners", corner_output)

    # Get the shapes in a given threshold image
    # Input -> Threshold image
    # Output-> list of  [x1,y1,x2,y2,area]
    def find_shapes(
        self,
        threshold_mat,
    ):
        contours, _ = cv2.findContours(
            threshold_mat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        print(f"contours{len(contours)}")
        # Get the co-ordinates and areas of each contour
        contour_properties = []
        for contour in contours:
            con_areas = cv2.contourArea(contour)
            print(f"Area{con_areas}")
            x, y, w, h = cv2.boundingRect(contour)
            row = [x, y, (x + w), (y + h), w, h, con_areas]
            contour_properties.append(row)
        return np.array(contour_properties).astype("intc")

    def getNormSaddle(self):
        # Get the saddle points of the board
        saddle = self.getSaddle(self.GRAY_IMG)
        saddle = -saddle
        saddle[saddle < 0] = 0
        self.pruneSaddle(saddle)

        # Normalize the saddle points
        saddle = (saddle - saddle.min()) / (saddle.max() - saddle.min())
        self.show_board("saddle", saddle)
        return saddle

    # Returns a mask based on contours
    def getContourMask(self, contour_properties):
        # Generate a contour mask for getting avg pixel values in the contour region
        mask = np.zeros_like(self.GRAY_IMG)
        for contour in contour_properties:
            print(contour)
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
    def resolve_contours(self, contour_properties):
        # find lower and upper range values
        areas = contour_properties[:, -1]
        median = np.median(areas)
        print(f"median ||||| {median}")
        indices = np.where((areas >= (0.15 * median)) & (areas <= (3 * median)))
        ret_contours = contour_properties[indices]
        return ret_contours

    def euclidieanDist(self, p1, p2):
        temp = p2 - p1
        return np.sqrt(np.dot(temp, temp.T))

    def draw_tiles(self, contours, img, n_contours=64):
        contour_properties = self.resolve_contours(contours)
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
        print(f"CENTROID ==>{centroid}")

        # Gather the dimensions to reduce to the required contours
        points = []
        dist_to_centroid = []
        for con in contour_properties:
            # cv2.rectangle(img,(con[0],con[1]),(con[2],con[3]),(255,0,0),2)
            x1, y1 = con[0], con[1]  # Top Left (x1,y1 )
            x2, y2 = con[0] + con[4], con[1]  # Top Right (x1+w,y1)
            x3, y3 = con[2], con[3]  # Bottom Right (x2,y2)
            x4, y4 = con[2] - con[4], con[3]  # Bottom Left (x2-w,y2 )
            distane_to_centroid = self.euclidieanDist(np.array([x1, y1]), centroid)

            points.append([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
            dist_to_centroid.append(distane_to_centroid)

        points = np.array(points)
        dist_to_centroid = np.array(dist_to_centroid)

        # find 32 contours(Dark tiles) that are the nearest to the centroid
        # sorted_points = points[points[:,-1].argsort()][:n_contours]
        sorted_points = points[dist_to_centroid.argsort()][:n_contours]

        # Extract the location of light tiles based on the location of dark tiles
        contour_mask = self.getContourMask(sorted_points)
        self.show_board("Contour Mask", contour_mask)
        # Image morphings for easier recognition of basic image features
        element = cv2.getStructuringElement(1, (5, 5), (3, 3))
        contour_mask = cv2.dilate(contour_mask, element)
        self.show_board("dilated_contour_mask", contour_mask)
        contour_mask = cv2.blur(contour_mask, (9, 9), 0)
        self.show_board("blured_dilated_contour_mask", contour_mask)

        print(f"====>>len(sorted_points){len(sorted_points)}")
        print(f"====>>sorted_points.shape{sorted_points.shape}")

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

        image = cv2.GaussianBlur(image, (5, 5), 0)
        self.show_board("NA - Gaussian Blur", image)

        image = self.auto_canny(image)
        self.show_board("NA - Canny", image)

        # image = cv2.dilate(image, (9, 9), 0)
        # self.show_board("NA - Dilated edges", image)

        image = cv2.GaussianBlur(image, (9, 9), 0)
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
            print(thresh)
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

    # Calibrates the board before the game starts
    # Needs an empty board to map the tiles on to 2D space
    # this space will be used as spatial reference throughout the game
    def calibrateBoard(self, img):
        # Detect the dark tiles on the board
        print("Board Calibration Satred")
        img, sorted_contours = self.detect_tiles(img)

        if len(sorted_contours) == 64:
            print(f"{len(sorted_contours)} tiles detected")
            return img, sorted_contours
        else:
            print(f"{len(sorted_contours)} tiles detected")
            return None, None

    def findBoardReference(self):
        img, contours = self.calibrateBoard(self.SRC_IMG.copy())
        if contours is not None:
            print("Board Found!")
            # Find the centroids of the tiles and sort based on Y-axis

            tile_centroids = contours.mean(axis=1).astype("int")
            sorted_coords = sorted(tile_centroids, key=lambda x: (x[1]))
            board = np.zeros((8, 8, 2), dtype="int")
            # Sort each row of 8 on X-axis
            for i in np.arange(0, 72, 8):
                print(f"Before sorted_coords[i:i+8]{sorted_coords[i:i+8]}")
                sorted_coords[i : i + 8] = sorted(
                    sorted_coords[i : i + 8], key=lambda x: (x[0])
                )
                print(f"After sorted_coords[i:i+8]{sorted_coords[i:i+8]}")

            board = np.array(sorted_coords, dtype="int").reshape(8, 8, 2)

            for i, tile in enumerate(sorted_coords):
                print(f"i = {i} , cord= {tile}")
                # print(f"tile[xy]{tile['xy']}")
                cv2.putText(
                    img,
                    str(i),
                    tile,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.25,
                    (255, 255, 255),
                    1,
                )

            self.show_board("numbered tiles", img)
            return board
        else:
            print("Please adjust the camera angle and recalibrate the board")

    def getBoardState(self):
        self.findBoardReference()
