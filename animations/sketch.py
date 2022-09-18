import cv2

# open an image using opencv
camera = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('../Upload/Anima4.avi', fourcc, 20.0, (640, 480))
# get image height and width

while True:
    _, img_gray = camera.read()
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    # img_blur = cv2.GaussianBlur(img_gray, (23, 23), 0, 0)
    # img_blend = cv2.divide(img_gray, img_blur, scale=200)
    cv2.imshow('@ElBruno - Pencil Sketch', img_gray)
    out.write(img_gray)
    if cv2.waitKey(5) & 0xFF == 27:
      break

out.release()
cv2.destroyAllWindows()
