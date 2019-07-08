import numpy as np
import cv2
import scipy.cluster.hierarchy as hcluster

def frame_extraction(video):
    # video = cv2.VideoCapture("./VideoPackage1/{}".format(video_name))
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame = []
    count = 0
    success,image = video.read()
    frame.append(image)
    diff_list = []
    threshold = 2000
    while success:
        if count < length-1:
            count+=1
            success,image = video.read()
            frame.append(image)
            value = np.sum(np.divide(cv2.absdiff(cv2.threshold(cv2.cvtColor(frame[count-1],cv2.COLOR_BGR2GRAY),127,255,cv2.THRESH_BINARY)[1],\
                        cv2.threshold(cv2.cvtColor(frame[count],cv2.COLOR_BGR2GRAY),127,255,cv2.THRESH_BINARY)[1]),255.0))
            if (value < threshold) and count>length/3:
                return image,count
            diff_list.append(value)
        else:
            return frame[np.argmin(diff_list)],np.argmin(diff_list)
        
def preprocessing(image):
    ## converting to grey scale
    img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
#     #histogram equalisation
#     img = cv2.equalizeHist(img)
    
    ## Guassian Blur
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    ## Bilateral Filter
    img = cv2.bilateralFilter(img,7,50,50)

    ## Auto canny 
    v = np.median(img)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    img = cv2.Canny(img, lower, upper)
    
    ## dilation and erode
    kernel = np.ones((3,3),np.uint8)
    img = cv2.dilate(img,kernel,iterations = 1)
    kernel = np.ones((3,3),np.uint8)
    img = cv2.erode(img,kernel,iterations = 1)
    
    return img

def blob_detection(img):
    params = cv2.SimpleBlobDetector_Params()                
    params.filterByArea = True
    params.filterByCircularity = True
    params.filterByInertia = True
    params.filterByConvexity = False
    
    params.minThreshold = 10
#     params.maxThreshold = 230
    params.minArea = 50
    params.maxArea = 220
    params.minCircularity = 0.30
    params.minInertiaRatio = 0.40
    params.minConvexity = 0.2
    detector = cv2.SimpleBlobDetector_create(params)       
    keypoints = detector.detect(img)  
    return keypoints

def cluster_data(data):
    thresh = 42.0
    clusters = hcluster.fclusterdata(data, thresh, criterion="distance")
    return clusters  


def Roi(orig_image):
    black_image = np.zeros(np.shape(orig_image),np.uint8)
    largest_area= 100000
    largest_contour_index=0
    img = cv2.cvtColor(orig_image,cv2.COLOR_BGR2GRAY)
    # Guassian Blur
    img = cv2.GaussianBlur(img, (5, 5), 0)
    ## Bilateral Filter
    img = cv2.bilateralFilter(img,7,50,50)
    v = np.median(img)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    img = cv2.Canny(img, 10, upper)
        
    kernel = np.ones((11,11),np.uint8)
    img = cv2.dilate(img,kernel,iterations = 2)
    
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(img,1,np.pi/180,100,minLineLength,maxLineGap)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        distance = np.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
        if distance > 350:
            cv2.line(black_image,(x1,y1),(x2,y2),(0,255,0),2)
    
    img = cv2.cvtColor(black_image,cv2.COLOR_BGR2GRAY)
    v = np.median(img)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    img = cv2.Canny(img, 10, upper)
    kernel = np.ones((13,13),np.uint8)
    img = cv2.dilate(img,kernel,iterations = 2)
    contours, _ = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = 0,0,np.shape(orig_image)[1],np.shape(orig_image)[0]
    for i,c in enumerate(contours):
        if cv2.contourArea(c) > largest_area:
            largest_area = cv2.contourArea(c)
            largest_contour_index = i
            rect = cv2.boundingRect(c)
            x,y,w,h = rect
    return x,y,w,h

def check_pos_keypoint(x1,y1,x2,y2,key_point):
    if (key_point.pt[0] < x2) and (key_point.pt[0] > x1) and (key_point.pt[1] < y2) and (key_point.pt[1] > y1):
        return True
    else:
        return False
    

def detect_dice(video_capture):
    
    detected_dice = []
    detected_dice_value = {}
    
    orig_image,reference_frame_no = frame_extraction(video_capture)
    image = preprocessing(orig_image)
    keypoints = blob_detection(image)
    data = [list(keypoint.pt) for keypoint in keypoints]
    clusters = cluster_data(data)
    keypoint_dict = {i:[] for i in np.unique(clusters)}
    
    im_with_keypoints = cv2.drawKeypoints(orig_image, keypoints, np.array([]), (0, 0, 255),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    x,y,w,h = Roi(orig_image)  
    for i in np.unique(clusters):
        for j in np.where(clusters==i)[0]:
            if (check_pos_keypoint(x,y,x+w,y+h,keypoints[j])):
                keypoint_dict[i].append(keypoints[j].pt)
            else:
                pass
        if len(keypoint_dict[i]) != 0:
            pos = tuple(np.mean(keypoint_dict[i],axis = 0,dtype=int))
            text = str(len(keypoint_dict[i]))
            
            dice_pos_value = list(np.mean(keypoint_dict[i],axis = 0,dtype=int))
            dice_pos_value.append(len(keypoint_dict[i]))
            detected_dice.append(dice_pos_value)

            # im_with_keypoints = write_text(pos,text,im_with_keypoints)
    return reference_frame_no,detected_dice

video_capture = cv2.VideoCapture("./video/2018-10-08@16-03-10.avi")
reference_frame_no, detected_dice = detect_dice(video_capture)

