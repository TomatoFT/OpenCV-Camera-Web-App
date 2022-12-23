import cv2
import itertools
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
from animations.landmark_detection import *
# Initialize the VideoCapture object to read from the webcam.
# camera_video = cv2.VideoCapture(0)
# camera_video.set(3,1280)
# camera_video.set(4,960)

def MU_effect(frame):
    # Read the left and right eyes images.
    left_eye = cv2.imread('Photo/MU.jpg')
    right_eye = cv2.imread('Photo/MU.jpg')
    # backgr = cv2.VideoCapture('Video/Fire_Wall_2.mp4')
    # Initialize the VideoCapture object to read from the smoke animation video stored in the disk.
    smoke_animation = cv2.VideoCapture('Video/fire.mp4')
    # Set the smoke animation video frame counter to zero.
    smoke_frame_counter = 0
    # Read a frame from smoke animation video
    _, smoke_frame = smoke_animation.read()
    # _, back = backgr.read()
    # Increment the smoke animation video frame counter.
    smoke_frame_counter += 1
    # Check if the current frame is the last frame of the smoke animation video.
    if smoke_frame_counter == smoke_animation.get(cv2.CAP_PROP_FRAME_COUNT):     
        # Set the current frame position to first frame to restart the video.
        smoke_animation.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # Set the smoke animation video frame counter to zero.
        smoke_frame_counter = 0
    # Flip the frame horizontally for natural (selfie-view) visualization.
    # frame = cv2.flip(frame, 1)
    # Perform Face landmarks detection.
    _, face_mesh_results = detectFacialLandmarks(frame, face_mesh_videos, display=False)
    # Check if facial landmarks are found.
    if face_mesh_results.multi_face_landmarks:
        # Get the mouth isOpen status of the person in the frame.
        _, mouth_status = isOpen(frame, face_mesh_results, 'MOUTH', 
                                     threshold=15, display=False)
        # Get the left eye isOpen status of the person in the frame.
        _, left_eye_status = isOpen(frame, face_mesh_results, 'LEFT EYE', 
                                        threshold=4.5 , display=False)
        # Get the right eye isOpen status of the person in the frame.
        _, right_eye_status = isOpen(frame, face_mesh_results, 'RIGHT EYE', 
                                         threshold=4.5, display=False)
        # Iterate over the found faces.
        for face_num, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
            # Check if the left eye of the face is open.
            if left_eye_status[face_num] == 'OPEN':  
                # Overlay the left eye image on the frame at the appropriate location.
                frame = overlay(frame, left_eye, face_landmarks,
                                'LEFT EYE', mp_face_mesh.FACEMESH_LEFT_EYE, display=False)
            # Check if the right eye of the face is open.
            if right_eye_status[face_num] == 'OPEN':
                # Overlay the right eye image on the frame at the appropriate location.
                frame = overlay(frame, right_eye, face_landmarks,
                                'RIGHT EYE', mp_face_mesh.FACEMESH_RIGHT_EYE, display=False)
            # Check if the mouth of the face is open.
            if mouth_status[face_num] == 'OPEN':
                # Overlay the smoke animation on the frame at the appropriate location.
                frame = overlay(frame, smoke_frame, face_landmarks, 
                                'MOUTH', mp_face_mesh.FACEMESH_LIPS, display=False)
    return frame


def Barca_effect(frame):
    # Read the left and right eyes images.
    left_eye = cv2.imread('Photo/Barca.png')
    right_eye = cv2.imread('Photo/Barca.png')
    # backgr = cv2.VideoCapture('Video/Fire_Wall_2.mp4')
    # Initialize the VideoCapture object to read from the smoke animation video stored in the disk.
    smoke_animation = cv2.VideoCapture('Video/fire.mp4')
    # Set the smoke animation video frame counter to zero.
    smoke_frame_counter = 0
    # Read a frame from smoke animation video
    _, smoke_frame = smoke_animation.read()
    # _, back = backgr.read()
    # Increment the smoke animation video frame counter.
    smoke_frame_counter += 1
    # Check if the current frame is the last frame of the smoke animation video.
    if smoke_frame_counter == smoke_animation.get(cv2.CAP_PROP_FRAME_COUNT):     
        # Set the current frame position to first frame to restart the video.
        smoke_animation.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # Set the smoke animation video frame counter to zero.
        smoke_frame_counter = 0
    # Flip the frame horizontally for natural (selfie-view) visualization.
    # frame = cv2.flip(frame, 1)
    # Perform Face landmarks detection.
    _, face_mesh_results = detectFacialLandmarks(frame, face_mesh_videos, display=False)
    # Check if facial landmarks are found.
    if face_mesh_results.multi_face_landmarks:
        # Get the mouth isOpen status of the person in the frame.
        _, mouth_status = isOpen(frame, face_mesh_results, 'MOUTH', 
                                     threshold=15, display=False)
        # Get the left eye isOpen status of the person in the frame.
        _, left_eye_status = isOpen(frame, face_mesh_results, 'LEFT EYE', 
                                        threshold=4.5 , display=False)
        # Get the right eye isOpen status of the person in the frame.
        _, right_eye_status = isOpen(frame, face_mesh_results, 'RIGHT EYE', 
                                         threshold=4.5, display=False)
        # Iterate over the found faces.
        for face_num, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
            # Check if the left eye of the face is open.
            if left_eye_status[face_num] == 'OPEN':  
                # Overlay the left eye image on the frame at the appropriate location.
                frame = overlay(frame, left_eye, face_landmarks,
                                'LEFT EYE', mp_face_mesh.FACEMESH_LEFT_EYE, display=False)
            # Check if the right eye of the face is open.
            if right_eye_status[face_num] == 'OPEN':
                # Overlay the right eye image on the frame at the appropriate location.
                frame = overlay(frame, right_eye, face_landmarks,
                                'RIGHT EYE', mp_face_mesh.FACEMESH_RIGHT_EYE, display=False)
            # Check if the mouth of the face is open.
            if mouth_status[face_num] == 'OPEN':
                # Overlay the smoke animation on the frame at the appropriate location.
                frame = overlay(frame, smoke_frame, face_landmarks, 
                                'MOUTH', mp_face_mesh.FACEMESH_LIPS, display=False)
    return frame

# Iterate until the webcam is accessed successfully.
# while camera_video.isOpened():
#     # Read a frame.
#     ok, frame = camera_video.read()
#     # Check if frame is not read properly then continue to the next iteration to read
#     # the next frame.
#     if not ok:
#         continue
#     # # Read a frame from smoke animation video
#     # _, smoke_frame = smoke_animation.read()
#     # _, back = backgr.read()
#     # # Increment the smoke animation video frame counter.
#     # smoke_frame_counter += 1
    
#     # # Check if the current frame is the last frame of the smoke animation video.
#     # if smoke_frame_counter == smoke_animation.get(cv2.CAP_PROP_FRAME_COUNT):     
#     #     # Set the current frame position to first frame to restart the video.
#     #     smoke_animation.set(cv2.CAP_PROP_POS_FRAMES, 0)
#     #     # Set the smoke animation video frame counter to zero.
#     #     smoke_frame_counter = 0
#     # # Flip the frame horizontally for natural (selfie-view) visualization.
#     # frame = cv2.flip(frame, 1)
#     # # Perform Face landmarks detection.
#     # _, face_mesh_results = detectFacialLandmarks(frame, face_mesh_videos, display=False)
#     # # Check if facial landmarks are found.
#     # if face_mesh_results.multi_face_landmarks:
#     #     # Get the mouth isOpen status of the person in the frame.
#     #     _, mouth_status = isOpen(frame, face_mesh_results, 'MOUTH', 
#     #                                  threshold=15, display=False)
#     #     # Get the left eye isOpen status of the person in the frame.
#     #     _, left_eye_status = isOpen(frame, face_mesh_results, 'LEFT EYE', 
#     #                                     threshold=4.5 , display=False)
#     #     # Get the right eye isOpen status of the person in the frame.
#     #     _, right_eye_status = isOpen(frame, face_mesh_results, 'RIGHT EYE', 
#     #                                      threshold=4.5, display=False)
#     #     # Iterate over the found faces.
#     #     for face_num, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
#     #         # Check if the left eye of the face is open.
#     #         if left_eye_status[face_num] == 'OPEN':  
#     #             # Overlay the left eye image on the frame at the appropriate location.
#     #             frame = overlay(frame, left_eye, face_landmarks,
#     #                             'LEFT EYE', mp_face_mesh.FACEMESH_LEFT_EYE, display=False)
#     #         # Check if the right eye of the face is open.
#     #         if right_eye_status[face_num] == 'OPEN':
#     #             # Overlay the right eye image on the frame at the appropriate location.
#     #             frame = overlay(frame, right_eye, face_landmarks,
#     #                             'RIGHT EYE', mp_face_mesh.FACEMESH_RIGHT_EYE, display=False)
#     #         # Check if the mouth of the face is open.
#     #         if mouth_status[face_num] == 'OPEN':
#     #             # Overlay the smoke animation on the frame at the appropriate location.
#     #             frame = overlay(frame, smoke_frame, face_landmarks, 
#     #                             'MOUTH', mp_face_mesh.FACEMESH_LIPS, display=False)
#     #         frame = overlay(frame, left_eye, face_landmarks, "HEAD", mp_face_mesh.FACEMESH_FACE_OVAL, display=False)
#     # Display the frame.
#     Newframe = MU_effect(frame)
#     # frame, back = cv2.resize(frame, (500, 500)), cv2.resize(back, (500, 500))
#     # frame = cv2.bitwise_or(frame, back)
#     cv2.imshow('Face Filter', Newframe)
#     # Wait for 1ms. If a key is pressed, retreive the ASCII code of the key.
#     k = cv2.waitKey(1) & 0xFF    
#     # Check if 'ESC' is pressed and break the loop.
#     if(k == 27):
#         break
# # Release the VideoCapture Object and close the windows.                  
# camera_video.release()
# cv2.destroyAllWindows()
