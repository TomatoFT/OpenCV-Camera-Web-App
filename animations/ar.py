
import numpy as np
import cv2
import imutils

# function to detect ArUco Markers
def findArucoMarkers(img, markerSize = 6, totalMarkers=250, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(cv2.aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    
    
    #Load the dictionary that was used to generate the markers.
    arucoDict = cv2.aruco.Dictionary_get(key)
    
    # Initialize the detector parameters using default values
    arucoParam = cv2.aruco.DetectorParameters_create()
    
    # Detect the markers
    bboxs, ids, rejected = cv2.aruco.detectMarkers(gray, arucoDict, parameters = arucoParam)
    return bboxs, ids
    
# Superimposing the image on the aruco markers detected in the video 
imgH=480
imgW=640

video = cv2. VideoCapture(0)

ret, video_frame=video.read()
image = cv2.imread('Photo/1.jpg')
image = cv2.resize(image, (imgH, imgW))
print('Hello')
while(video.isOpened()):
    if ret==True:
        refPts=[]  
        #Detect the Aruco markers on the video frame
        arucofound =findArucoMarkers(video_frame, totalMarkers=100)
        h, w = video_frame.shape[:2]
        
        # if the aruco markers are detected
        if  len(arucofound[0])!=0:
                
                for Corner, id in zip(arucofound[0], arucofound[1]):
                    corners = Corner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))
                    # draw lines around the marker and display the marker id
                    cv2.line(video_frame, topLeft, topRight, (0, 255, 0), 2)
                    cv2.line(video_frame, topRight, bottomRight, (0, 255, 0), 2)
                    cv2.line(video_frame, bottomRight, bottomLeft, (0, 255, 0), 2)
                    cv2.line(video_frame, bottomLeft, topLeft, (0, 255, 0), 2)                    
                    cv2.putText(video_frame, str(id),(topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
                    corner = np.squeeze(Corner)
                    refPts.append(corner)
                    # only when all the 4 markes are detected in the image
                    if len(refPts)==4:
                        ( refPtBR, refPtTR,refPtBL, refPtTL) = refPts
                        video_pt = np.array([  refPtTL[3], refPtBL[3],refPtBR[2], refPtTR[3]])
                       
                        # grab the spatial dimensions of the  image and define the
                        # transform matrix for the image in 
                        #top-left, top-right,bottom-right, and bottom-left order
                        image_pt = np.float32([[0,0], [h,0], [h,w], [0,w]])
                        
                        # compute the homography matrix between the image and the video frame
                        matrix, _ = cv2.findHomography( image_pt, video_pt)
                        
                        #warp the  image to video frame based on the homography
                        warped  = cv2.warpPerspective(image, matrix, (video_frame.shape[1], video_frame.shape[0]))
                        
                        #Create a mask representing region to 
                        #copy from the warped image into the video frame.
                        mask = np.zeros((imgH, imgW), dtype="uint8")
                        cv2.fillConvexPoly(mask, video_pt.astype("int32"), (255, 255, 255),cv2.LINE_AA)
                                                                    
                        # give the source image a black border
                        # surrounding it when applied to the source image,
                        #you can apply a dilation operation
                        rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                        mask = cv2.dilate(mask, rect, iterations=2)
                        
                        # Copy the mask with the three channel version by stacking it depth-wise,
                        # This will allow copying the warped source image into the input image
                        maskScaled = mask.copy() / 255.0
                        maskScaled = np.dstack([maskScaled] * 3)
                        print('Hello')
                        # Copy the masked warped image into the video frame by
                        # (1) multiplying the warped image and masked together, 
                        # (2) multiplying the Video frame with the mask 
                        # (3) adding the resulting images
                        warpedMultiplied = cv2.multiply(warped.astype("float"), maskScaled)
                        imageMultiplied = cv2.multiply(video_frame.astype(float), 1.0 - maskScaled)
                        #imgout = video frame multipled with mask 
                        #        + warped image multipled with mask
                        output = cv2.add(warpedMultiplied, imageMultiplied)
                        output = output.astype("uint8")
                        cv2.imshow("output", output)
    
    ret, video_frame=video.read()
    key = cv2.waitKey(20)
    # if key q is pressed then break 
    if key == 113:
        break 
    
#finally destroy/close all open windows
video.release()
cv2.destroyAllWindows()