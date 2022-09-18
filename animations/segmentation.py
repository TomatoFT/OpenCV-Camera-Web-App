import cv2
import mediapipe as mp
import numpy as np

class Segmentation():

    def __init__(self, model=1):
        """
        :param model: model type 0 or 1. 0 is general 1 is landscape(faster)
        """
        self.model = model
        self.mpDraw = mp.solutions.drawing_utils
        self.mpSelfieSegmentation = mp.solutions.selfie_segmentation
        self.selfieSegmentation = self.mpSelfieSegmentation.SelfieSegmentation(self.model)

    def removeBG(self, img, imgBg=(255, 255, 255), threshold=0.1):
        """
        :param img: image to remove background from
        :param imgBg: BackGround Image
        :param threshold: higher = more cut, lower = less cut
        :return:
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.selfieSegmentation.process(imgRGB)
        condition = np.stack(
            (results.segmentation_mask,) * 3, axis=-1) > threshold
        if isinstance(imgBg, tuple):
            _imgBg = np.zeros(img.shape, dtype=np.uint8)
            _imgBg[:] = imgBg
            imgOut = np.where(condition, img, _imgBg)
        else:
            imgOut = np.where(condition, img, imgBg)
        return imgOut

# segmentor = Segmentation()
# img = cv2.imread("Photo/6.jpg")
# imgBg = cv2.imread("Photo/transparents.jpg")
# img, imgBg = cv2.resize(img, (500, 500)), cv2.resize(imgBg, (500, 500))
# imgOut = segmentor.removeBG(img, threshold=0.55)
# imgNew = cv2.bitwise_or(imgOut, imgBg)
# cv2.imshow("Image Out", imgNew)
# cv2.imwrite("QuanMU.png", imgNew)
# cv2.waitKey(0)
