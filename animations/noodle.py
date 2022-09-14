import cv2
import numpy as np
from time import perf_counter
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
scale_fact = 1;

segment_count = fps*3

segment_height = int(height*scale_fact/segment_count)

print("segment count:", segment_count, "\nscaling factor:", scale_fact, "\nsegment_height",segment_height)
frames = []
t1 = perf_counter()

while(cap.isOpened()):
    ret, new_frame = cap.read()
    if new_frame is None:
        break
    if scale_fact != 1:
        new_frame = cv2.resize(new_frame,
                               (int(new_frame.shape[1]*scale_fact),
                                int(new_frame.shape[0]*scale_fact)))
    frames.append(new_frame)
    if len(frames) >= segment_count:    
        segments = []
        for i,frame in enumerate(frames):
            segments.append(frame[i*segment_height:(i+1)*segment_height])

        noodled_frame = np.concatenate(segments, axis=0)

        frames.pop(0)
        cv2.imshow('frame', noodled_frame)
        t2 = perf_counter()
        delay = int(1000/fps - (t2-t1)*1000)
        delay = delay if delay >1 else 1
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
        t1 = perf_counter()
    # frame = noodled_effect(new_frame)
    # cv2.imshow("winname", frame)
    # frames.pop(0)
    # t2 = perf_counter()
    # delay = int(1000/fps - (t2-t1)*1000)
    # delay = delay if delay >1 else 1
    # if cv2.waitKey(delay) & 0xFF == ord('q'):
    #     break
    # t1 = perf_counter()
cap.release()
cv2.destroyAllWindows()