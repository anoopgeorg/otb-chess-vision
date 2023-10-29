import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from pathlib import Path
import ultralytics


class Chessboard():
    
    # initialize with the emty board image during calibration
    # Input -> Image , Weights
    def __init__(self,src_img,weights_path:str=None):
        self.SRC_IMG = src_img
        self.H       = src_img.shape[0]
        self.W       = src_img.shape[1]
        self.GRAY_IMG = cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)     
        self.WEIGHTS_PATH = Path(weights_path) if weights_path is not None else None  
    
    def show_baord(self,title,img):
        cv2.imshow(title,img)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
        
    
    # Calibrates the board before the game starts
    # Needs an empty board to map the tiles on to 2D space
    # this space will be used throughout the game
    def calibrate_board(self):
        # Get the corners of the board using YOLO model
        
        # 4 Corners detected 
            # Perspective transformation of the board
            
            
        # Not 4 corners raise message to adjust camera and re-caliberate
        pass
        

    
    
    def auto_canny(self,img,sigma=0.33):
        print("Edge extraction started")
        median = np.median(img)
        lower_bound = int(max(0,(1.0 - sigma)*median))
        upper_bound = int(max(0,(1.0 + sigma)*median))
        img_edges = cv2.Canny(img,lower_bound,upper_bound)
        return img_edges    
        
    def smoothen_image(self,img):
        img_c = img.copy() 
        img_c = cv2.blur(img_c,(9,9),0)
        self.show_baord("smooothen_blur",img_c)
        print(type(img_c))
        
        
        element = cv2.getStructuringElement(1, (7, 7),(3, 3))
        img_c = cv2.dilate(img_c,element)
        self.show_baord("dialated_blur",img_c)   
        
        img_c = cv2.cvtColor(img_c,cv2.COLOR_BGR2GRAY)
        
        return img_c
        
        
    def draw_points(self,img:np.array,points:np.array):
        if points is not None:    
            for point in points:
                x = point[0]
                y = point[1]
                plotted_img = cv2.circle(img, (x,y), radius=2, color=(255,0,0), thickness=1)
        
            self.show_baord("poi_img",plotted_img)
                
                
                
    def find_clusters(self,points):
        dbscan = DBSCAN(eps=2,min_samples=15)
        model = dbscan.fit(points)
        labels = model.labels_ 
        # Identify the grid points as the core samples
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[model.core_sample_indices_] = True
        grid_points = points[core_samples_mask]
        
        return grid_points

 
        
        
    def detect_corners(self):
        self.show_baord("gray_image",self.GRAY_IMG)
        
        manipulated_img = self.smoothen_image(self.SRC_IMG)
        self.show_baord("manipulated_img",manipulated_img)
        
        gray = np.float32(manipulated_img)
        # Detector parameters
        blockSize = 2
        apertureSize = 3
        k = 0.04
        
        dst = cv2.dilate(cv2.cornerHarris(gray,blockSize,apertureSize,k),None)
        self.show_baord("dst_img",dst)
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
        # self.show_baord("final_corners",corner_output)
        
        corner_output = self.SRC_IMG.copy()   
        criteria = dst>0.05*dst.max()
        corner_output[criteria]=[0,255,0]

        
        # poi = np.array(np.where(criteria)).T
        idx = np.where(criteria)
        # poi = self.find_clusters(np.array(list(zip(idx[1], idx[0]))))
        poi = np.array(list(zip(idx[1], idx[0])))
        self.draw_points(self.SRC_IMG,poi)
        self.show_baord("final_corners",corner_output)
    
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
    
    
    # Resolve the contour list to get only the required contours
    # Input -> List of [x1,y1,x2,y2,area]
    # Output-> list of [x1,y1,x2,y2,area]
    def resolve_contours(self,contour_properties):
        # find lower and upper range values
        areas = contour_properties[:,-1] 
        #normalized_area = (areas - areas.min()) / (areas.max()-areas.min())
        # print(f"len(contour_properties):  {len(normalized_area)}")
        # print(f"Standard deviation ==> {np.std(normalized_area)}")
        # print(f"mean ==> {normalized_area}")
        # print(f"max ==> {normalized_area}")
        # print(f"min ==> {normalized_area}")
        #q1,q2,q3 = np.percentile(areas,[25,50,75])
        # l,u = np.percentile(normalized_area,[25,95])
        #iqr = q3-q1
        #lower_bound = q1 - (1.5*iqr)
        #upper_bound = q3 + (1.5*iqr)
        # print(f"lower_bound,upper_bound : {lower_bound,upper_bound}")
        # print(f"l,u {l,u}")
        # print("=============")
        # print(areas.sort())
        
        median = np.median(areas)
        print(f"median ||||| {median}")
        indices = np.where((areas >= (0.15*median) )& (areas <= (3*median)))
        #indices = np.where((normalized_area >= lower_bound )& (normalized_area <= upper_bound))
        # #indices = np.where((areas >= l )& (areas <= u))
        # print(len(contour_properties[indices]))
        return contour_properties[indices]
        #return contour_properties
    
    def draw_tiles(self,contour_properties,img):
        for con in contour_properties:
            
            # cv2.rectangle(img,(con[0],con[1]),(con[2],con[3]),(255,0,0),2)
            t_left  = (con[0],con[1]) 
            t_right = (con[0]+con[4],con[1])
            b_right = (con[2],con[3])
            b_left  = (con[2]-con[4],con[3])
            
            cv2.circle(img,t_left,1,(255,0,0),3)
            cv2.circle(img,t_right,1,(0,255,0),3)
            cv2.circle(img,b_right,1,(0,0,255),3)
            cv2.circle(img,b_left,1,(0,0,0),3)
        return img
    
    
    def detect_tiles(self):
        img_c = self.SRC_IMG.copy()
        self.show_baord("OG_img",img_c)
        
        # Image morphings for easier recognition of basic image features
        element = cv2.getStructuringElement(1, (7, 7),(3, 3))
        img_c = cv2.dilate(img_c,element)
        self.show_baord("dilate",img_c)
        res_img = cv2.cvtColor(img_c,cv2.COLOR_BGR2GRAY)
        self.show_baord("gray_img",res_img)
        
        # Thresholding image to get binary and an inverse binary image
        _,thresh_img = cv2.threshold(res_img,150,255,cv2.THRESH_BINARY)
        self.show_baord("thershold_img",thresh_img)        
        _,thresh_inv = cv2.threshold(res_img,150,255,cv2.THRESH_BINARY_INV)
        self.show_baord("thresh_inv",thresh_inv)
        
        # Find the contours for the threshold image and the inverse threshold image
        # Using both threshold image and the inverse threshold gives better results
        thresh_contours = self.find_shapes(thresh_img)
        thresh_inv_contours = self.find_shapes(thresh_inv)
       
        contours_binary = self.resolve_contours(thresh_contours)
        return_img = self.SRC_IMG.copy()
        return_img = self.draw_tiles(contours_binary,return_img)
        self.show_baord("final_detection",return_img)
        
                        # cv2.rectangle(return_img,(x,y),(x+w,y+h),(255,0,0),2)
                        
                        
                        # self.show_baord("final_detection",return_img)
                        