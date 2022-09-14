import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

def landmarks_list(IMAGE):
  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_face_mesh = mp.solutions.face_mesh

  drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
  with mp_face_mesh.FaceMesh(
      static_image_mode=True,
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.5) as face_mesh:

    ##### INPUT IMAGE HERE:
    IMAGE_gray = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2GRAY)

    image = IMAGE
    image_gray = IMAGE_gray.copy()
    ##### INPUT IMAGE HERE:
    
    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    landmarks_points = []
    annotated_image = image.copy()
    mask_mp = np.zeros_like(image_gray)

    # Obtiene los puntos en formato [x,y] desde los landmark normalizados.
    for face_landmarks in results.multi_face_landmarks:
      for n in range(0, 468):
        x = int(face_landmarks.landmark[n].x * image.shape[1])
        y = int(face_landmarks.landmark[n].y * image.shape[0])
        landmarks_points.append((x, y))


    # Dibuja las figuras sobre el face mesh.
      mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())

  return [landmarks_points, annotated_image]

def convexhull_face(landmarks_list, IMAGE):

  image = IMAGE.copy()

  # Creación de una figura convexa a partir de los puntos más externos
  points = np.array(landmarks_list, np.int32)
  convexhull = cv2.convexHull(points)

  # Superposición de la figura sobre la imagen
  cv2.polylines(image, [convexhull], True, (255, 0, 0), 2)

  return [convexhull, image]

def segment_face(convexhull, IMAGE):

  image = IMAGE.copy()
  image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  mask_mp = np.zeros_like(image_gray)

  # Crea una máscara a partir de convexhull sobre la imagen negra mask_mp.
  cv2.fillConvexPoly(mask_mp, convexhull, 255)

  # Aplica un AND sobre la imagen original y la máscara.
  face_seg = cv2.bitwise_and(image, image, mask = mask_mp)

  return [mask_mp, face_seg]

def triangulation(convexhull, landmarks, IMAGE):

  image = IMAGE.copy()
  points = np.array(landmarks, np.int32)
  triangle_indxs = []

  # Rectangulo que contiene a la cara.
  rect = cv2.boundingRect(convexhull)

  # Define el sector a triangular y realiza la triangulación de puntos.
  t_sector = cv2.Subdiv2D(rect)
  t_sector.insert(landmarks)
  triangles = t_sector.getTriangleList()
  triangles = np.array(triangles, dtype=np.int32)

  # Plot rectangulo.
  (x,y,w,h) = rect
  cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

  # Plot triangulos.
  max_x = x + w 
  max_y = y + h

  for p in triangles:
    pt1 = (p[0], p[1])
    pt2 = (p[2], p[3])
    pt3 = (p[4], p[5])

    indx_pt1 = np.where((points == pt1).all(axis=1))[0][0]
    indx_pt2 = np.where((points == pt2).all(axis=1))[0][0]
    indx_pt3 = np.where((points == pt3).all(axis=1))[0][0]
    
    triangulo = [indx_pt1, indx_pt2, indx_pt3]
    triangle_indxs.append(triangulo)
    
    # print(indx_pt1)

    # if (max_x, max_y) >= pt1 and pt2 and pt3 >= (x, y):
    cv2.line(image, pt1, pt2, (255, 0, 255))
    cv2.line(image, pt2, pt3, (255, 0, 255)) 
    cv2.line(image, pt1, pt3, (255, 0, 255))

  return [triangles, triangle_indxs, image]

def  triangulation_second(convexhull, landmarks_2, triangle_indxs, IMAGE):
  image = IMAGE.copy()
  points = np.array(landmarks_2, np.int32)
  triangles2_coord = []

  rect = cv2.boundingRect(convexhull)
  (x,y,w,h) = rect
  cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

  for indxs in triangle_indxs:
    pt1 = landmarks_2[indxs[0]]
    pt2 = landmarks_2[indxs[1]]
    pt3 = landmarks_2[indxs[2]]

    triangles2_coord.append([pt1, pt2, pt3])

    cv2.line(image, pt1, pt2, (255, 0, 255))
    cv2.line(image, pt2, pt3, (255, 0, 255)) 
    cv2.line(image, pt1, pt3, (255, 0, 255))
  
  return [triangles2_coord, image]

def get_single_triangle(tr_indx, landmarks, IMAGE):
  image = IMAGE.copy()
  
  # Coordenadas originales de los vertices del triangulo.
  tr_pt1 = landmarks[tr_indx[0]]
  tr_pt2 = landmarks[tr_indx[1]]
  tr_pt3 = landmarks[tr_indx[2]]
  triangle = np.array([tr_pt1, tr_pt2, tr_pt3], np.int32)

  # Rectangulo que encierra al triangulo.
  rect = cv2.boundingRect(triangle)
  (x, y, w, h) = rect

  # Recorte de imágen y creación de máscara de un sólo triangulo.
  rect_single_tr = image[y: y + h, x: x + w]
  rect_single_tr_gray = cv2.cvtColor(rect_single_tr, cv2.COLOR_BGR2GRAY)
  rect_single_tr_mask = np.zeros_like(rect_single_tr_gray)

  # Coordenadas del triangulo a nivel de segmentación.
  points_segment = np.array([[tr_pt1[0] - x, tr_pt1[1] - y] , 
                              [tr_pt2[0] - x, tr_pt2[1] - y] , 
                              [tr_pt3[0] - x, tr_pt3[1] - y] ], np.int32)
  
  # Se rellena el triangulo con blanco, en un fondo negro y se aplica and lógico.
  cv2.fillConvexPoly(rect_single_tr_mask, points_segment, 255)
  segmented_triangle = cv2.bitwise_and(rect_single_tr, rect_single_tr, mask=rect_single_tr_mask)

  return [points_segment, rect, segmented_triangle, rect_single_tr]


def face_swap(triangle_indxs, landmarks_1, landmarks_2, mask_bin1, mask_bin2, IMAGE1, IMAGE2):
  image1 = IMAGE1.copy()
  image2 = IMAGE2.copy()

  image1_swap = np.zeros_like(image1)
  image2_swap = np.zeros_like(image2)

  # Se recorren todos los triangulos con formato índice.
  for tr_indx in triangle_indxs:
    tr1 = get_single_triangle(tr_indx, landmarks_1, image1)
    tr2 = get_single_triangle(tr_indx, landmarks_2, image2)

    (x2, y2, w2, h2) = (tr2[1][0], tr2[1][1], tr2[1][2], tr2[1][3])

    # Se obtienen los puntos de los vértices de cada triangulo.
    tr1_points = np.float32(tr1[0])
    tr2_points = np.float32(tr2[0])

    cropped_tr2_mask = np.zeros((h2, w2), np.uint8)
    cv2.fillConvexPoly(cropped_tr2_mask, tr2[0], 255)

    # Métrica para warping y warping de triangulos, para transformación 1-2.
    metrics1 = cv2.getAffineTransform(tr1_points, tr2_points)
    warped_tr1 = cv2.warpAffine(tr1[3], metrics1, (w2, h2))
    warped_tr1 = cv2.bitwise_and(warped_tr1, warped_tr1, mask=cropped_tr2_mask)

    # Transformación cara 1-2.
    triangle_area1 = image2_swap[y2: y2 + h2, x2: x2 + w2]
    triangle_area1_gray = cv2.cvtColor(triangle_area1, cv2.COLOR_BGR2GRAY)
    _, mask_triangles_designed = cv2.threshold(triangle_area1_gray, 1, 255, cv2.THRESH_BINARY_INV)
    warped_tr1 = cv2.bitwise_and(warped_tr1, warped_tr1, mask=mask_triangles_designed)

    triangle_area1 = cv2.add(triangle_area1, warped_tr1)
    image2_swap[y2: y2 + h2, x2: x2 + w2] = triangle_area1
  
  # Reemplazo de cara 1-2.
  image2_wout_face = cv2.bitwise_and(image2, image2, mask = np.invert(mask_bin2))
  image2_replace = cv2.add(image2_wout_face, image2_swap)

  # (x_r1, y_r1, w_r1, h_r1) = cv2.boundingRect(mask_bin2)
  # center_face2 = (int((x_r1 + x_r1 + w_r1) / 2), int((y_r1 + y_r1 + h_r1) / 2))
  # image2_replace = cv2.seamlessClone(image2_replace, image2, mask_bin2, center_face2, cv2.MIXED_CLONE)

  return image2_replace

X1 = cv2.VideoCapture(0)

X2 = cv2.VideoCapture('Video/dancing.mp4')
X3 = cv2.VideoCapture('Video/Fire_Wall_2.mp4')
while True:
  _, X1_frame = X1.read()
  _, X2_frame = X2.read()
  _, X3_frame = X3.read()
  X1_frame, X2_frame, X3_frame = cv2.resize(X1_frame, (500, 500)), cv2.resize(X2_frame, (500, 500)), cv2.resize(X3_frame, (500, 500))
  X1_gray = cv2.cvtColor(X1_frame, cv2.COLOR_BGR2GRAY)
  X2_gray = cv2.cvtColor(X2_frame, cv2.COLOR_BGR2GRAY)
  # img = X2_frame.copy()
  img_gray = X2_gray.copy()
  mask = np.zeros_like(img_gray)

  landmarks_X2 = landmarks_list(X2_frame)
  convexhull_X2 = convexhull_face(landmarks_X2[0], X2_frame)
  face_seg_X2 = segment_face(convexhull_X2[0], X2_frame)
  triangulacion_X2 = triangulation(convexhull_X2[0], landmarks_X2[0], convexhull_X2[1])
  
  landmarks_X1 = landmarks_list(X1_frame)
  convexhull_X1 = convexhull_face(landmarks_X1[0], X1_frame)
  face_seg_X1 = segment_face(convexhull_X1[0], X1_frame)
  triangulacion_X1 = triangulation_second(convexhull_X1[0], landmarks_X1[0], triangulacion_X2[1], convexhull_X1[1])
  
  swap1 = face_swap(triangulacion_X2[1], landmarks_X2[0], landmarks_X1[0], face_seg_X2[0], face_seg_X1[0], X2_frame, X1_frame)
  swap2 = face_swap(triangulacion_X2[1], landmarks_X1[0], landmarks_X2[0], face_seg_X1[0], face_seg_X2[0], X1_frame, X2_frame)

  swap1 = cv2.bitwise_or(X3_frame, swap1, mask=None)
  cv2.imshow("aa",swap1)
  cv2.imshow("ss",swap2)

  if cv2.waitKey(5) & 0xFF == 27:
    break

cv2.destroyAllWindows()
X1_frame.release()