import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from pathlib import Path
from ultralytics import YOLO


class Chessboard():
    
    # initialize with the emty board image during calibration
    # Input -> Image , Weights
    def __init__(self,src_img,weights_path:str=None): 
        self.SRC_IMG  = cv2.resize(src_img,(416,416))
        # self.SRC_IMG  = src_img
        self.H        = self.SRC_IMG.shape[0]
        self.W        = self.SRC_IMG.shape[1]
        self.GRAY_IMG = cv2.cvtColor(self.SRC_IMG,cv2.COLOR_BGR2GRAY)
        # self.WEIGHTS_PATH = Path(weights_path) if weights_path is not None else None
        # self.CORNER_DETECTOR = self.load_model() 
         
    # def load_model(self):
    #     if self.WEIGHTS_PATH is not None:
    #         try:
    #             model = YOLO(str(self.WEIGHTS_PATH))
    #             model.fuse()
    #         except Exception as e:
    #             print(e)
    #     return model
    
    
    def show_board(self,title,img):
        pass
        # cv2.imshow(title,img)
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows()
    
    def find_anchor_points(self,xys):
        anchor = np.zeros((2,2,2))
        temp = xys.numpy().copy()
        x_sorted = temp[:,0].argsort()
        
        # Get the Left and Right side anchor points
        left_anchors = temp[x_sorted[:2]]
        right_anchors = temp[x_sorted[2:]]
        
        # Determine the top and bottom corners
        top_left = left_anchors[left_anchors[:,1].argmax()]
        bottom_left = left_anchors[left_anchors[:,1].argmin()]
        
        top_right = right_anchors[right_anchors[:,1].argmax()]
        bottom_right = right_anchors[right_anchors[:,1].argmin()]
        
        print(top_left)
        print(top_right)
        print(bottom_left)
        print(bottom_right)
        # Fill the anchor with the corners based on thier locations
        anchor[0,0],anchor[1,0],anchor[0,1],anchor[1,1]=top_left,bottom_left,top_right,bottom_right
        return anchor
    
    # Calibrates the board before the game starts
    # Needs an empty board to map the tiles on to 2D space
    # this space will be used throughout the game
    # def calibrate_board(self,img):
    #     pass
        # Get the corners of the board using YOLO model
        # if self.CORNER_DETECTOR is not None:
        #     try:
        # self.show_board("Input Image",self.SRC_IMG)
        
        
        
        # corners_xy = results[0].boxes.xywh[:,0:2]
        # if len(corners_xy) == 4:
        #     # Find the anchor points for image transformations
        #     anchors = self.find_anchor_points(corners_xy)
        #     self.draw_points(self.SRC_IMG,anchors.flatten())
        #     # Perspective transformation of the board
            
        # else:
        #     print(f"Number of corners detected : , Need to deetect 4 \
        #                                     corners{len(corners_xy)}")
            
                
            # except Exception as e:
            #     print(e)
            #     print("Boad calibration failed")
                

            
            
        # Not 4 corners raise message to adjust camera and re-caliberate
        
        

    
    
    def auto_canny(self,img,sigma=0.33):
        print("Edge extraction started")
        median = np.median(img)
        lower_bound = int(max(0,(1.0 - sigma)*median))
        upper_bound = int(max(0,(1.0 + sigma)*median))
        img_edges = cv2.Canny(img,lower_bound,upper_bound)
        self.show_board("Canny Edge",img_edges)
        return img_edges    
        
    def smoothen_image(self,img):
        img_c = img.copy() 
        img_c = cv2.blur(img_c,(9,9),0)
        self.show_board("smooothen_blur",img_c)
        print(type(img_c))
        
        
        element = cv2.getStructuringElement(1, (7, 7),(3, 3))
        img_c = cv2.dilate(img_c,element)
        self.show_board("dialated_blur",img_c)   
        
        img_c = cv2.cvtColor(img_c,cv2.COLOR_BGR2GRAY)
        
        return img_c
        
        
    def draw_points(self,img:np.array,points:np.array):
        print(f"draw_points{points}")
        if points is not None:    
            for point in points:
                x = point[0]
                y = point[1]
                plotted_img = cv2.circle(img, (x,y), radius=2, color=(255,0,0), thickness=1)
        
            self.show_board("poi_img",plotted_img)
                
                
                
    def find_clusters(self,points):
        
        dbscan = DBSCAN(eps=40,min_samples=3)
        model = dbscan.fit(points)
        labels = model.labels_ 
        print(f"<+++++++++++++++++>")
        print(f"Clusters Found{len(labels)}")
        print(labels)
        
        

 
    def detect_good_features(self):        
        manipulated_img = self.GRAY_IMG
        element = cv2.getStructuringElement(1, (9, 9),(3, 3))
        manipulated_img = cv2.dilate(manipulated_img,element)
        self.show_board("dilate",manipulated_img)
        manipulated_img = cv2.cvtColor(manipulated_img,cv2.COLOR_BGR2GRAY)
        self.show_board("gray_img",manipulated_img)
        
        corners = cv2.goodFeaturesToTrack(manipulated_img,maxCorners=300,qualityLevel=0.2,minDistance=25,useHarrisDetector=True,k=0.1)
        corners = np.uint8(corners)
        img_c = self.SRC_IMG.copy()
        for c in corners:
            x,y = c.ravel()
            img_c = cv2.circle(img_c,(x,y),3,(255,0,0),-1)
        self.show_board("goodfeatures",img_c)
        
        
    def detect_corners(self):
        self.show_board("gray_image",self.GRAY_IMG)
        
        manipulated_img = self.smoothen_image(self.SRC_IMG)
        self.show_board("manipulated_img",manipulated_img)
        
        gray = np.float32(manipulated_img)
        # Detector parameters
        blockSize = 2
        apertureSize = 3
        k = 0.04
        
        dst = cv2.dilate(cv2.cornerHarris(gray,blockSize,apertureSize,k),None)
        self.show_board("dst_img",dst)
        ret,dst = cv2.threshold(dst,0.01*dst.max(),255,0)
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
        criteria = dst>0.05*dst.max()
        corner_output[criteria]=[0,255,0]

        
        # poi = np.array(np.where(criteria)).T
        idx = np.where(criteria)
        # poi = self.find_clusters(np.array(list(zip(idx[1], idx[0]))))
        poi = np.array(list(zip(idx[1], idx[0])))
        self.draw_points(self.SRC_IMG,poi)
        self.show_board("final_corners",corner_output)
    
    # Get the shapes in a given threshold image
    # Input -> Threshold image
    # Output-> list of  [x1,y1,x2,y2,area]
    def find_shapes(self,threshold_mat,):
        contours,_ = cv2.findContours(threshold_mat,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"contours{len(contours)}")  
        # Get the co-ordinates and areas of each contour      
        contour_properties = []
        for contour in contours:
            con_areas = cv2.contourArea(contour)
            print(f"Area{con_areas}")
            x,y,w,h = cv2.boundingRect(contour)
            row = [x,y,(x+w),(y+h),w,h,con_areas]
            contour_properties.append(row)             
        return np.array(contour_properties).astype("intc")
    
    def getNormSaddle(self):
         # Get the saddle points of the board
        saddle = self.getSaddle(self.GRAY_IMG) 
        saddle = -saddle
        saddle[saddle<0] = 0
        self.pruneSaddle(saddle)
        
        # Normalize the saddle points
        saddle = (saddle - saddle.min()) / (saddle.max() - saddle.min())
        self.show_board("saddle",saddle)
        return saddle
    
    # Returns a mask based on contours
    def getContourMask(self,contour_properties):
        # Generate a contour mask for getting avg pixel values in the contour region 
        mask = np.zeros_like(self.GRAY_IMG)
        for contour in contour_properties:
            print(contour)
            cv2.rectangle(mask,(contour[0],contour[1]),(contour[2],contour[3]),(255,255,255),thickness=-1)
            
        return mask      
    
    # Identofy the contours that most likely form a grid
    def getMostLikelyGrid(self,contour_properties):
        # print("Finding the most likely Grid")
        # top_left_pnts = contour_properties[:,0:2]
        # print() 
        pass
    
    
    # Resolve the contour list to get only the required contours
    # Input -> List of [x1,y1,x2,y2,x3,y3,x4,y4,w,h,area]
    # Output-> list of [x1,y1,x2,y2,x3,y3,x4,y4,w,h,area]
    def resolve_contours(self,contour_properties):
        # find lower and upper range values
        areas = contour_properties[:,-1]         
        median = np.median(areas)
        print(f"median ||||| {median}")
        indices = np.where((areas >= (0.15*median) )& (areas <= (3*median)))
        ret_contours = contour_properties[indices] 
       
        contour_mask = self.getContourMask(ret_contours)
        self.show_board("Contour Mask",contour_mask)
        
        return ret_contours
    
    def euclidieanDist(self,p1,p2):
        temp = p2 - p1
        return np.sqrt(np.dot(temp,temp.T))
    
    def draw_tiles(self,contours,img,n_contours=32):
        contour_properties = self.resolve_contours(contours)
        # Centroid of all the contour points should ideally be the middle of the chessboard        
        t_left  = np.array([contour_properties[:,0],contour_properties[:,1]])                          # x1,y1
        t_right = np.array([contour_properties[:,0]+contour_properties[:,4],contour_properties[:,1]])  # x1+w,y1
        b_right = np.array([contour_properties[:,2],contour_properties[:,3]])                          # x2,y2
        b_left  = np.array([contour_properties[:,2]-contour_properties[:,4],contour_properties[:,3]])  # x2-w,y2
        t_left_m  = t_left.mean(axis=1)
        t_right_m = t_right.mean(axis=1)
        b_right_m = b_right.mean(axis=1)
        b_left_m  = b_left.mean(axis=1)
        centroid = np.mean([t_left_m,t_right_m,b_right_m,b_left_m],axis=0).astype("int")
        print(f"CENTROID ==>{centroid}")

        # Gather the dimensions to reduce the required contours 
        points = []        
        for con in contour_properties:
            
            # cv2.rectangle(img,(con[0],con[1]),(con[2],con[3]),(255,0,0),2)
            x1,y1  =  con[0],con[1]                  # Top Left (x1,y1 )
            x2,y2  =  con[0]+con[4],con[1]           # Top Right (x1+w,y1)
            x3,y3  =  con[2],con[3]                  # Bottom Right (x2,y2)
            x4,y4  =  con[2]-con[4],con[3]           # Bottom Left (x2-w,y2 )
            distane_to_centroid = self.euclidieanDist(np.array([x1,y1]),centroid)
            points.append([x1,y1,x2,y2,x3,y3,x4,y4,distane_to_centroid])       
        points = np.array(points)

        
        # find 32 contours that are the nearest to the centroid 
        sorted_points = points[points[:,-1].argsort()][:n_contours]
        print(f"====>>len(sorted_points){len(sorted_points)}")
        print(f"====>>sorted_points.shape{sorted_points.shape}")
        
        # Plot the detected contours
        for point in sorted_points:
            print((point[0],point[1]))
            cv2.circle(img,(int(point[0]),int(point[1])),1,(255,0,0),3)
            cv2.circle(img,(int(point[2]),int(point[3])),1,(0,255,0),3)
            cv2.circle(img,(int(point[4]),int(point[5])),1,(0,0,255),3)
            cv2.circle(img,(int(point[6]),int(point[7])),1,(0,0,0),3)        
        cv2.circle(img,centroid,1,(128,0,128),5)
        return img
    
    
    def detect_tiles(self):
        img_c = self.SRC_IMG.copy()
        self.show_board("OG_img",img_c)
                
        # Image morphings for easier recognition of basic image features
        element = cv2.getStructuringElement(1, (9, 9),(3, 3))
        img_c = cv2.dilate(img_c,element)
        self.show_board("dilate",img_c)
        
        res_img = cv2.cvtColor(img_c,cv2.COLOR_BGR2GRAY)
        self.show_board("gray_img",res_img)        
        res_img = cv2.blur(res_img,(7,7),0)
        self.show_board("blur",res_img)

        # Create threshold masks for contour detection
        thresh_img = cv2.adaptiveThreshold(res_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,101,25)
        self.show_board("thershold_img",thresh_img) 
        # thresh_inv = cv2.adaptiveThreshold(res_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,101,25)
        # self.show_board("thershold_img",thresh_img)          

        
        # Find the contours for the threshold image 
        thresh_contours = self.find_shapes(thresh_img)
        # thresh_inv_contours = self.find_shapes(thresh_inv)
       
       
        
        # Get the normalized saddle points
        # saddle_mask = self.getNormSaddle()
        
        
        return_img = self.SRC_IMG.copy()
        return_img = self.draw_tiles(thresh_contours,return_img)
        self.show_board("final_detection",return_img)
        return return_img
        
                        # cv2.rectangle(return_img,(x,y),(x+w,y+h),(255,0,0),2)
                        
                        
                        # self.show_board("final_detection",return_img)
    
    def pruneSaddle(self,s):
        thresh = 128
        score = (s>0).sum()
        while (score > 10000):
            print(thresh)
            thresh = thresh*2
            s[s<thresh] = 0
            score = (s>0).sum()
    
    
    def getSaddle(self,gray_img):
        img = gray_img.astype(np.float64)
        gx = cv2.Sobel(img,cv2.CV_64F,1,0)
        gy = cv2.Sobel(img,cv2.CV_64F,0,1)
        gxx = cv2.Sobel(gx,cv2.CV_64F,1,0)
        gyy = cv2.Sobel(gy,cv2.CV_64F,0,1)
        gxy = cv2.Sobel(gx,cv2.CV_64F,0,1)
        
        S = gxx*gyy - gxy**2
        return S
    


    
    # Calibrates the board before the game starts
    # Needs an empty board to map the tiles on to 2D space
    # this space will be used as spatial reference throughout the game
    def calibrate_board(self):
        img_c = self.GRAY_IMG.copy()
        # Image morphings for easier recognition of basic image features
        element = cv2.getStructuringElement(1, (9, 9),(3, 3))
        img_c = cv2.dilate(img_c,element)
        self.show_board("dilate",img_c)        
        img_c = cv2.blur(img_c,(3,3),0)
        self.show_board("blur",img_c)
        
      
        # Thresholding image to get binary and an inverse binary image        
        thresh_img = cv2.adaptiveThreshold(img_c,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,101,25)
        self.show_board("thershold_img",thresh_img)        
        thresh_inv = cv2.adaptiveThreshold(img_c,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,101,25)
        self.show_board("thresh_inv",thresh_inv)
        
        print("Mask and Connected components")
        cc = cv2.connectedComponentsWithStats(thresh_inv,8,cv2.CV_32S)
        
        (tot_labels,label_ids,values,centroid) = cc
        
        coef = 1
        boxes = []
        for i in range(1,tot_labels):
            x1 = int(values[i,cv2.CC_STAT_LEFT]*coef)
            y1 = int(values[i,cv2.CC_STAT_TOP]*coef)
            w = int(values[i,cv2.CC_STAT_WIDTH]*coef)
            h = int(values[i,cv2.CC_STAT_HEIGHT]*coef)
            area = values[i,cv2.CC_STAT_AREA]*coef          
            boxes.append([x1,y1,w,h,area])
        cc_overlay = self.SRC_IMG.copy()
        if boxes is not None:
            areas = boxes[-1]
            print(f"np.median(area){np.median(area)}")
            plt.hist(areas,bins=100)
            plt.show()
            for box in boxes:
                x1,y1,w,h,area = box
                print(area)
                cv2.rectangle(cc_overlay,(x1,y1),(x1+w,y1+h),(255,0,0),2)
            self.show_board("CC_OVERLAY",cc_overlay)
        else:
            print("No CC found")
        
        
        
        # Get the saddle points of the board
        saddle = self.getSaddle(self.GRAY_IMG) 
        saddle = -saddle
        saddle[saddle<0] = 0
        self.pruneSaddle(saddle)
        # saddle[saddle<0.5*saddle.max()] = 0
        self.show_board("saddle",saddle)
        plt.imshow(saddle)
        plt.show()
        print(f"saddle.min(){saddle.min()}")
        print(f"saddle.max(){saddle.max()}")
        
    
        