import cv2

# open an image using opencv
imgOriginal = cv2.VideoCapture(0)


# get image height and width

while True:
    _, img_gray = imgOriginal.read()
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (23, 23), 0, 0)
    img_blend = cv2.divide(img_gray, img_blur, scale=200)
    cv2.imshow('@ElBruno - Pencil Sketch', img_blend)
    if cv2.waitKey(5) & 0xFF == 27:
      break

# # save image using opencv
# cv2.imwrite('OfficePencilSketch.jpg', img)

# key controller
cv2.waitKey(0)
cv2.destroyAllWindows()
img_gray.release()