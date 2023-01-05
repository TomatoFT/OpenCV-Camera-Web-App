import cv2
import mediapipe as mp
import numpy    as np
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import numpy as np
# import imageio
# from PIL import Image, ImageSequence


frame_counter = 0
frame_counter_horn = 0
frame_counter_heart = 0
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# glass = cv2.imread('glasses/gl3-removebg.png', -1)
# glass[:,:,0]= 0
# glass[:,:,1]= 0
# glass[:,:,2]= 250


cap = cv2.VideoCapture(0)

cap1 = cv2.VideoCapture('example.mp4')

cap2 = cv2.VideoCapture('video/fire_up.webm')

cap_horn = cv2.VideoCapture('video/horn.webm')

cap_heart = cv2.VideoCapture('video/break.webm')

# declare to write mp4 file
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output3.mp4', fourcc, 20.0, (640,480))


def transparent2(img, targetImg, x, y, alpha=0.8):
    newFrame = img.copy()
    try: 
      size = (int(targetImg.shape[1]*alpha), int(targetImg.shape[0]*alpha))
      targetImg = cv2.resize(targetImg, size)

      b, g, r = cv2.split(targetImg)
      overlay_color = cv2.merge((b, g, r)) 

      img2gray = cv2.cvtColor(targetImg,cv2.COLOR_BGR2GRAY)
      ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
      h, w, _ = overlay_color.shape
      roi = newFrame[y:y + h, x:x + w]

      img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
      img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)
      newFrame[y:y + h, x:x + w] = cv2.add(img1_bg, img2_fg)
    except:
      pass
    return newFrame

def transparent(img, targetImg, x, y, size= (300,300)):
    if size is not None:
        targetImg = cv2.resize(targetImg, size)

    newFrame = img.copy()
    b, g, r, a = cv2.split(targetImg)
    overlay_color = cv2.merge((b, g, r)) 
    mask = cv2.medianBlur(a, 1)
    h, w, _ = overlay_color.shape
    roi = newFrame[y:y + h, x:x + w]

    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)
    newFrame[y:y + h, x:x + w] = cv2.add(img1_bg, img2_fg)

    return newFrame

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


face_mesh =  mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

hands =  mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
  
while cap.isOpened() or cap1.isOpened() or cap2.isOpened():

  success, image = cap.read()
  okay1  , frame1 = cap1.read()
  okay1  , frame2 = cap2.read()
  okay1  , frame_horn = cap_horn.read()
  okay1  , frame_heart = cap_heart.read()

  # frame1[:,:,0]=0
  # frame1[:,:,1]=0
  if not success:
    print("Ignoring empty camera frame.")
    # If loading a video, use 'break' instead of 'continue'.
    continue

  # loop effect
  frame_counter += 1
  if frame_counter == cap2.get(cv2.CAP_PROP_FRAME_COUNT):
    frame_counter = 0 #Or whatever as long as it is the same as next line
    cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
  
  frame_counter_horn += 1
  if frame_counter_horn == cap_horn.get(cv2.CAP_PROP_FRAME_COUNT):
    frame_counter_horn = 0 #Or whatever as long as it is the same as next line
    cap_horn.set(cv2.CAP_PROP_POS_FRAMES, 0)
  
  frame_counter_heart += 1
  if frame_counter_heart == cap_heart.get(cv2.CAP_PROP_FRAME_COUNT):
    frame_counter_heart = 0 #Or whatever as long as it is the same as next line
    cap_heart.set(cv2.CAP_PROP_POS_FRAMES, 0)
  

  # To improve performance, optionally mark the image as not writeable to
  # pass by reference.

  image.flags.writeable = False
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  results_hands = hands.process(image)
  results = face_mesh.process(image)

  # Draw the face mesh annotations on the image.
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


  if results_hands.multi_hand_landmarks:
      for hand_landmarks in results_hands.multi_hand_landmarks:
        # mp_drawing.draw_landmarks(
        #     image,
        #     hand_landmarks,
        #     mp_hands.HAND_CONNECTIONS,
        #     mp_drawing_styles.get_default_hand_landmarks_style(),
        #     mp_drawing_styles.get_default_hand_connections_style())
        
        image_rows, image_cols, _ = image.shape
        
        count = _normalized_to_pixel_coordinates(hand_landmarks.landmark[8].x,hand_landmarks.landmark[8].y,
                                              image_cols,image_rows)


        image = transparent2(image, frame1, count[0]-110, count[1]-160,0.5)

        # cv2.putText(image, "Good Night!!", count, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255,0.4), 3,cv2.LINE_AA)


#=============================================================================
  if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
      

      image_rows, image_cols, _ = image.shape
      

      coor1 = _normalized_to_pixel_coordinates(face_landmarks.landmark[7].x,face_landmarks.landmark[7].y,
                                          image_cols,image_rows)
      # image = transparent(image, glass, coor1[0]-30, coor1[1]-70, (150,150))

      coor2 = _normalized_to_pixel_coordinates(face_landmarks.landmark[6].x,face_landmarks.landmark[6].y,
                                             image_cols,image_rows)

      image = transparent2(image, frame2, coor2[0]-160, coor2[1] - 220,0.5)

      image = transparent2(image, frame_horn, coor2[0]-225, coor2[1] - 220,0.7)

      # image = transparent2(image, frame_heart, 200 , 200 ,0.5)

      # cv2.circle(image,count, 63, (0,0,255), -1)
      # image = cv2.flip(image, 1)
      out.write(image)
      
  
  # Flip the image horizontally for a selfie-view display.
  cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
  if cv2.waitKey(5) & 0xFF == 27:
    break
cap.release()
cap1.release()