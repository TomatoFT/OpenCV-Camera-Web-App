import cv2 as cv
import numpy as np 

blank = np.zeros(shape=(500,500,3), 
                 dtype='uint8')
# blank[0:250] = 255, 0, 255
# blank[250:500] = 0 , 255, 100
# # print(blank)
# cv.imshow('blank', blank)
cv.line(blank, (10,12), (40,90), color=(255,0,255), thickness=2)
cv.imshow("line", blank)
cv.putText(blank, "Fuck you", (255,255), fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(255,255,0),thickness=3)
cv.imshow("winname", blank)
cv.waitKey(0)
