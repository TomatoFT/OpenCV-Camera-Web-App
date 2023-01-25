import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from scipy.ndimage import binary_erosion
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageEnhance
import random
import imageio
# import argparse
# import array
# import cairo
# from PIL import ImageFilter, Image
# from tools import transform_color
cam = cv.VideoCapture(0)
fourcc = cv.VideoWriter_fourcc(*'XVID')
#out = cv.VideoWriter('test.mp4', fourcc, 10.0, (640,480))
# initialize Pose estimator

# Làm mờ đen 4 góc
def darkfour(img):
    rows, cols = img.shape[:2]

    # generating vignette mask using Gaussian kernels
    kernel_x = cv.getGaussianKernel(cols, 800)
    kernel_y = cv.getGaussianKernel(rows, 255)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    output = np.copy(img)

    # applying the mask to each channel in the input image
    for i in range(3):
        output[:, :, i] = output[:, :, i] * mask
    return output

# Tạo tia neon bắn ra từ 2 mắt 
def neon_eye(img, eye1, eye2, lenx, leny):
    img_neon = np.zeros((img.shape[0],img.shape[1],1))

    img_neon = cv.line(img_neon, (eye1[0]+ lenx, eye1[1]+ leny), (eye1[0] + lenx+30, eye1[1] + leny+30), (255,255,255), 5)
    img_neon = cv.line(img_neon, (eye2[0]+lenx, eye2[1]+leny), (eye2[0] + lenx+30, eye2[1] + leny+30), (255,255,255), 5)

    imarray = np.asarray(img_neon)[..., 0] / 255

    eroded = binary_erosion(imarray, iterations=3)

    # Make the outlined rectangles.
    outlines = imarray - eroded

    # Convolve with a Gaussian to effect a blur.
    blur = gaussian_filter(outlines, sigma=3)

    # Make binary images into neon green.
    # neon_green_rgb = [0.124, 1, 0.0784]
    neon_green_rgb = [0, 0, 1]
    outlines = outlines[:, :, None] * neon_green_rgb
    blur = blur[:, :, None] * neon_green_rgb

    # Combine the images and constrain to [0, 1].
    blur_strength = 3
    glow = np.clip(outlines + blur_strength*blur, 0, 1)
    glow = glow*255
    output = np.where(glow < np.array([50, 50, 50]), img, glow)
    return output

# Tạo khung viền bằng neon 
def neon_background(img):
    img_neon = np.zeros((img.shape[0],img.shape[1],1))

    img_neon = cv.line(img_neon, (30, 30), (30, img_neon.shape[0]-30), (255,255,255), 10)
    img_neon = cv.line(img_neon, (30, 30), (img_neon.shape[1]-30, 30), (255,255,255), 10)
    img_neon = cv.line(img_neon, (img_neon.shape[1]-30,30), (img_neon.shape[1]-30, img_neon.shape[0]-30), (255,255,255), 10)
    img_neon = cv.line(img_neon, (30, img_neon.shape[0]-30), (img_neon.shape[1]-30, img_neon.shape[0]-30), (255,255,255), 10)
    # img_neon = cv.circle(img_neon, (int(img.shape[1]/2),int(img.shape[0]/2)), int(img.shape[1]/2-40), (255,255,255), -1)
    imarray = np.asarray(img_neon)[..., 0] / 255

    eroded = binary_erosion(imarray, iterations=3)

    outlines = imarray - eroded

    blur = gaussian_filter(outlines, sigma=5)

    neon_green_rgb = [0, 1, 0]
    outlines = outlines[:, :, None] * neon_green_rgb
    blur = blur[:, :, None] * neon_green_rgb

    blur_strength = 3
    glow = np.clip(outlines + blur_strength*blur, 0, 1)
    glow = glow*255
    output = np.where(glow < np.array([50, 50, 50]), img, glow)
    output = cv.line(output, (30, 30), (30, output.shape[0]-30), (255,255,255), 10)
    output = cv.line(output, (30, 30), (output.shape[1]-30, 30), (255,255,255), 10)
    output = cv.line(output, (output.shape[1]-30,30), (output.shape[1]-30, output.shape[0]-30), (255,255,255), 10)
    output = cv.line(output, (30, output.shape[0]-30), (output.shape[1]-30, output.shape[0]-30), (255,255,255), 10)
    # output = cv.circle(output, (int(output.shape[1]/2),int(output.shape[0]/2)), int(output.shape[1]/2-40), (255,255,255), 10)
    return output

def hand(targetImg, x, y, size = None):
    if size is not None:
        targetImg = cv.resize(targetImg, size)

    newFrame = frame.copy()
    b, g, r, a = cv.split(targetImg)
    overlay_color = cv.merge((b, g, r))
    mask = cv.medianBlur(a, 1)
    h, w, _ = overlay_color.shape
    roi = newFrame[y:y + h, x:x + w]

    img1_bg = cv.bitwise_and(roi.copy(), roi.copy(), mask=cv.bitwise_not(mask))
    img2_fg = cv.bitwise_and(overlay_color, overlay_color, mask=mask)
    newFrame[y:y + h, x:x + w] = cv.add(img1_bg, img2_fg)

    return newFrame

# Khoảng cách giữa 2 điểm 
def caculate_distance(x1, y1, x2, y2):
    a = (x1-x2)*(x1-x2)*1.0
    b = (y1-y2)*(y1-y2)*1.0
    c = math.sqrt(a+b+0.001)
    return c

# Tạo vòng tròn ma thuật 
def circle_magic(image,img_1, img_2, x1, y1, deg, diameter):
    h,w,c =image.shape
    if x1 < 0:
        x1 = 0
    elif x1 > w:
        x1 = w
    if y1 < 0:
        y1 = 0
    elif y1 > h:
        y1 = h
    if x1 + diameter > w:
        diameter = w - x1
    if y1 + diameter > h:
        diameter = h - y1

    shield_size = diameter, diameter
    hei, wid, col = img_1.shape
    cen = (wid // 2, hei // 2)
    M1 = cv.getRotationMatrix2D(cen, round(deg), 1.0)
    M2 = cv.getRotationMatrix2D(cen, round(360 - deg), 1.0)
    rotated1 = cv.warpAffine(img_1, M1, (wid, hei))
    rotated2 = cv.warpAffine(img_2, M2, (wid, hei))
    if (diameter != 0):
        image = hand(rotated1, x1, y1, shield_size)
        # image = hand(rotated2, x1, y1, shield_size)
    deg = deg + 5.0
    if deg > 360:
        deg = 0
    return image, deg


def draw_devil(image, lanmark, matna, size = 100, x = 100, y = 100):
    # try: 
    #     count_5 = _normalized_to_pixel_coordinates(lanmark.landmark[5].x,lanmark.landmark[5].y,image.shape[0],image.shape[1])
    #     img_neon = np.zeros((image.shape[0],image.shape[1],3))
    #     matna = cv.resize(matna, (int(7*size/10),size))
    #     img_neon[count_5[0]:count_5[0]+matna.shape[0],count_5[1]:count_5[1]+matna.shape[1]] = matna
    #     # print(str(x) + " " + str(y) + " "+ str(x+matna.shape[0]) + " " + str(y+matna.shape[1]))
    #     output = np.where(img_neon < np.array([10, 10, 10]), image, img_neon)
    # except:
    img_neon = np.zeros((image.shape[0],image.shape[1],3))
    matna = cv.resize(matna, (int(7*size/10),size))
    img_neon[x:x+matna.shape[0],y:y+matna.shape[1]] = matna
    # print(str(x) + " " + str(y) + " "+ str(x+matna.shape[0]) + " " + str(y+matna.shape[1]))
    output = np.where(img_neon < np.array([10, 10, 10]), image, img_neon)
    # pass
    return output

def devil_eye(img, count_5, count_2):
    img_neon = np.zeros((img.shape[0],img.shape[1],1))
    x = random.randint(0,1000)
    img_neon = cv.line(img_neon, (count_5[0],count_5[1]), (x,img_neon.shape[1]-20), (255,0,0), 5)
    img_neon = cv.line(img_neon, (count_2[0],count_2[1]), (x+60,img_neon.shape[1]-20), (255,0,0), 5)

    imarray = np.asarray(img_neon)[..., 0] / 255

    eroded = binary_erosion(imarray, iterations=3)

    # Make the outlined rectangles.
    outlines = imarray - eroded

    # Convolve with a Gaussian to effect a blur.
    blur = gaussian_filter(outlines, sigma=3)

    # Make binary images into neon green.
    # neon_green_rgb = [0.124, 1, 0.0784]
    neon_green_rgb = [0, 0, 1]
    outlines = outlines[:, :, None] * neon_green_rgb
    blur = blur[:, :, None] * neon_green_rgb

    # Combine the images and constrain to [0, 1].
    blur_strength = 3
    glow = np.clip(outlines + blur_strength*blur, 0, 1)
    glow = glow*255
    output = np.where(glow < np.array([50, 50, 50]), img, glow)
    # output = cv.line(output, (count_5[0],count_5[1]), (x,output.shape[1]-20), (255,255,0), 5)
    # output = cv.line(output, (count_2[0],count_2[1]), (x+60,output.shape[1]-20), (255,255,0), 5)
    return output

def DK_circle_magic(frame,magic_circle_cww,magic_circle_cw, lanmark, deg, earthquake,count_earthquake, frame_cols, frame_rows):
    try:
        count_19 = _normalized_to_pixel_coordinates(lanmark.landmark[19].x,lanmark.landmark[19].y,frame_cols,frame_rows)
        count_20 = _normalized_to_pixel_coordinates(lanmark.landmark[20].x,lanmark.landmark[20].y,frame_cols,frame_rows)
        count_12 = _normalized_to_pixel_coordinates(lanmark.landmark[12].x,lanmark.landmark[12].y,frame_cols,frame_rows)
        count_11 = _normalized_to_pixel_coordinates(lanmark.landmark[11].x,lanmark.landmark[11].y,frame_cols,frame_rows)
        kc19_20 = caculate_distance(count_19[0], count_19[1], count_20[0], count_20[1])
        kc11_12 = caculate_distance(count_11[0], count_11[1], count_12[0], count_12[1])
        diameter = int(10*kc19_20)

        x1 = count_20[0]
        y1 = count_20[1]
        if count_20[1] > count_12[1] and kc19_20 < 100:
            # frame, deg = circle_magic(frame,magic_circle_cww, magic_circle_cw, count_19[0], count_19[1],deg, diameter)
            frame, deg = circle_magic(frame,magic_circle_cww, magic_circle_cw, count_20[0]-100, count_20[1],deg, diameter)
            if count_earthquake % 3 and count_earthquake < 40:
                earthquake[:, :-50] = frame[:,50:]
                frame = earthquake
            if count_earthquake % 4 and count_earthquake < 40:
                earthquake[:-20,:] = frame[20:,:]
                frame = earthquake
            count_earthquake = count_earthquake + 1
    except:
        pass
    return frame, deg, count_earthquake

def DK_neon_eye(frame, lanmark, frame_cols, frame_rows):
    try:
        #print(_normalized_to_pixel_coordinates(b.landmark[9].x,b.landmark[9].y,frame_cols,frame_rows))
        count_2 = _normalized_to_pixel_coordinates(lanmark.landmark[2].x,lanmark.landmark[2].y,frame_cols,frame_rows)
        count_5 = _normalized_to_pixel_coordinates(lanmark.landmark[5].x,lanmark.landmark[5].y,frame_cols,frame_rows)
        count_7 = _normalized_to_pixel_coordinates(lanmark.landmark[7].x,lanmark.landmark[7].y,frame_cols,frame_rows)
        count_8 = _normalized_to_pixel_coordinates(lanmark.landmark[8].x,lanmark.landmark[8].y,frame_cols,frame_rows)
        count_19 = _normalized_to_pixel_coordinates(lanmark.landmark[19].x,lanmark.landmark[19].y,frame_cols,frame_rows)
        count_20 = _normalized_to_pixel_coordinates(lanmark.landmark[20].x,lanmark.landmark[20].y,frame_cols,frame_rows)
        kc7_8 = caculate_distance(count_7[0], count_7[1], count_8[0], count_8[1])
        kc7_19 = caculate_distance(count_7[0], count_7[1], count_19[0], count_19[1])
        kc8_20 = caculate_distance(count_8[0], count_8[1], count_20[0], count_20[1])
        if (kc7_19*1.0/ kc7_8 > 0.7 and kc7_19*1.0/ kc7_8 < 1 ) or (kc8_20*1.0/ kc7_8 > 0.7 and kc8_20*1.0/ kc7_8 < 1 ):
            frame = neon_eye(frame, count_5, count_2, lenx, leny)
            if lenx > 400:
                lenx = 0
            if leny > 400:
                leny = 0
            lenx = lenx + 5
            leny = leny + 5
        # kc7_19 = math.sqrt((count_7[0]- count_19[0])*(count_7[0]- count_19[0]) + (count_7[1]- count_19[1])*(count_7[1]- count_19[1]))
        # kc8_20 = math.sqrt((count_20[0]- count_8[0])*(count_20[0]- count_8[0]) + (count_20[1]- count_8[1])*(count_20[1]- count_8[1]))
    except:
        pass
    return frame

def DK_draw_devil(frame, devil_mask, lanmark, frame_cols, frame_rows, count_devil, count_trang, count_earthquake_def):
    # hopden = np.zeros((500,500,3))
    if count_earthquake_def < 50:
        return frame, count_devil, count_trang
    try:
        count_20 = _normalized_to_pixel_coordinates(lanmark.landmark[20].x,lanmark.landmark[20].y,frame_cols,frame_rows)
        count_19 = _normalized_to_pixel_coordinates(lanmark.landmark[19].x,lanmark.landmark[19].y,frame_cols,frame_rows)
        count_5 = _normalized_to_pixel_coordinates(lanmark.landmark[5].x,lanmark.landmark[5].y,frame_cols,frame_rows)
        count_2 = _normalized_to_pixel_coordinates(lanmark.landmark[2].x,lanmark.landmark[2].y,frame_cols,frame_rows)
        count_8 = _normalized_to_pixel_coordinates(lanmark.landmark[8].x,lanmark.landmark[8].y,frame_cols,frame_rows)
        count_7 = _normalized_to_pixel_coordinates(lanmark.landmark[7].x,lanmark.landmark[7].y,frame_cols,frame_rows)
        count_10 = _normalized_to_pixel_coordinates(lanmark.landmark[10].x,lanmark.landmark[10].y,frame_cols,frame_rows)
        count_12 = _normalized_to_pixel_coordinates(lanmark.landmark[12].x,lanmark.landmark[12].y,frame_cols,frame_rows)
        count_9 = _normalized_to_pixel_coordinates(lanmark.landmark[9].x,lanmark.landmark[9].y,frame_cols,frame_rows)
        # count_12 = _normalized_to_pixel_coordinates(lanmark.landmark[12].x,lanmark.landmark[12].y,frame_cols,frame_rows)

        kc205 = caculate_distance(count_20[0], count_20[1], count_5[0], count_5[1])
        kc192 = caculate_distance(count_19[0], count_19[1], count_2[0], count_2[1])
        kc2010 = caculate_distance(count_20[0], count_20[1], count_10[0], count_10[1])
        kc87 = caculate_distance(count_8[0], count_8[1], count_7[0], count_7[1])
        kc105 = caculate_distance(count_5[0], count_5[1], count_10[0], count_10[1])
        # if kcmatna <100:
        # frame = draw_devil(frame, hopden, devil_mask, 150, count_20[0], count_20[1])
        if kc205 < 100 and kc192 < 100:
            frame = cv.circle(frame, (count_5[0], count_5[1]), 10, (255,0,0), -1)
            frame = cv.circle(frame, (count_2[0], count_2[1]), 10, (255,0,0), -1)
            frame = devil_eye(frame, count_5, count_2)
        # print(str(count_5[0]))
        if count_devil == 0:
            frame = draw_devil(frame, lanmark, devil_mask, int(kc87*2.2), count_8[1]-int(kc105*2), count_8[0]-int(kc105/1.8))
            if count_trang % 3 and count_trang < 20:
                frame = np.ones((frame.shape[0], frame.shape[1], 3))
                frame = frame*255
            count_trang = count_trang + 1
            return frame, count_devil, count_trang
        if kc2010 > 60:
            frame = draw_devil(frame, lanmark, devil_mask, 150, count_20[1]-150, count_20[0])
        else:
            if count_20[0] > count_12[0] and count_20[1] > count_12[1]:
                return frame, count_devil, count_trang
            count_devil = 0 

    except:
        # frame = draw_devil(frame, lanmark, devil_mask,100, 60, 160)
        # frame = frame
        pass 
    return frame, count_devil, count_trang

# fourcc = cv.VideoWriter_fourcc(*'MP4V')
# out = cv.VideoWriter('output3.mp4', fourcc, 20.0, (640,480))

# main
if __name__ == '__main__':
    
    # cam.set(cv.CAP_PROP_FRAME_WIDTH, 1000)
    # cam.set(cv.CAP_PROP_FRAME_HEIGHT, 500)
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    mp_drawing_styles = mp.solutions.drawing_styles
    mp_selfie_segmentation = mp.solutions.selfie_segmentation

    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    deg = 0 
    magic_circle_cww = cv.imread('D:/Github/TIKTOK_effect_CS231_UIT/image/magic_2.png', -1)
    magic_circle_cw = cv.imread('D:/Github/TIKTOK_effect_CS231_UIT/image/magic_circle_cw.png', -1)
    dagiac = cv.imread('D:/Github/TIKTOK_effect_CS231_UIT/image/KQdagiac.jpg',-1)
    devil_mask = cv.imread('D:/Github/TIKTOK_effect_CS231_UIT/image/KQmatna.jpg')
    hopden = np.zeros((500,500,3))
    count_earthquake = 1
    count_earthquake_def = 1
    count_devil = 1
    lenx = 0
    leny = 0
    count_trang = 0
    # fourcc = cv.VideoWriter_fourcc(*'XVID')
    # out = cv.VideoWriter('output.mp4', fourcc, 20.0, (960, 540))
    while cam.isOpened():
        ret,frame = cam.read()
        frame.flags.writeable = False
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results_pose = pose.process(frame)
        frame.flags.writeable = True
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        a = mp_drawing_styles.get_default_pose_landmarks_style()
        lanmarks = results_pose.pose_landmarks
        c = mp_pose.POSE_CONNECTIONS
        # mp_drawing.draw_landmarks(frame, lanmarks, c, landmark_drawing_spec=a)
        earthquake = np.zeros((frame.shape[0],frame.shape[1],3))
        # frame, deg, lenx, leny = poses(frame,b, deg, lenx, leny, count_dongdat)
        frame_rows, frame_cols, _ = frame.shape 
        frame = darkfour(frame)
        if count_earthquake_def < 80:
            frame, deg, count_earthquake_def = DK_circle_magic(frame,magic_circle_cww, magic_circle_cww, lanmarks, deg, earthquake, count_earthquake, frame_cols, frame_rows)
        # frame = DK_neon_eye(frame, lanmarks, frame_cols, frame_rows)
            count_earthquake = count_earthquake_def
        if count_earthquake_def % 3 and count_earthquake_def < 80 and count_earthquake_def >= 70:
                frame = np.zeros((frame.shape[0], frame.shape[1], 3))
                # frame = frame*255
        if count_earthquake_def >= 80:
            frame, count_devil_def, count_trang_def = DK_draw_devil(frame, devil_mask, lanmarks, frame_cols, frame_rows, count_devil, count_trang, count_earthquake_def)
            count_devil = count_devil_def
            count_trang = count_trang_def
        if cv.waitKey(1) ==ord('q'):
            break
        frame = frame[:, :-50]
        frame = frame[:-20, :]
        frame = neon_background(frame)
        # cv.imshow('TIKTOK', frame)
        cv.imwrite('D:/Github/TIKTOK_effect_CS231_UIT/KQ_TIKTOK.png', frame)
        # out.write(frame)

    cam.release()
    cv.destroyAllWindows()
    # duc_foreground = cv.imread('D:/DATASCIENTIST/CS231/Buoi09/people.jpg')
    # frame = poses(duc_foreground)
    # convexHull(frame)
    # cv.imshow('Output',frame)
    # cv.imwrite('../result_img.jpg', frame)

