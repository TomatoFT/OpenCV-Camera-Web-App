import cv2
import numpy as np

# cap = cv2.VideoCapture(0)
img = np.zeros((500, 1080, 3), np.uint8)
i = 0
def tiktok_animation(frame, speed=700, i=0):   
    frame = cv2.resize(frame ,(1080, 500))
    # frame = np.fliplr(frame)
    h, w = frame.shape[:2]
    img[:,i+speed:w,:] = frame[:, i+speed:w, :]
    cv2.line(img, (i+ speed, 0), (i+speed, h), (0,255,0), 2)
    img[:,i:i+speed, :] = frame[:, i:i+speed, :]
    frame = cv2.flip(frame,1)
    return img
# while True:
#     _, frame = cap.read()
#     img = tiktok_animation(frame, 1, i)
#     cv2.imshow("frame", img)
#     i +=1
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# cv2.destroyAllWindows()
# cap.release()