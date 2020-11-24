import cv2
import numpy as np


def empty(a):
    pass


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


path = "../dataset/1.png"  # '../Resources/lena.png'
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("GaussianBlur_ksize", "TrackBars", 2, 15, empty)
cv2.createTrackbar("GaussianBlur_SigmaX", "TrackBars", 1, 20, empty)
cv2.createTrackbar("canny_Th1", "TrackBars", 10, 150, empty)
cv2.createTrackbar("canny_Th2", "TrackBars", 70, 255, empty)
# cv2.createTrackbar("Val Min","TrackBars",153,255,empty)
# cv2.createTrackbar("Val Max","TrackBars",255,255,empty)

img = cv2.imread(path)
img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.namedWindow("Stacked Images", cv2.WINDOW_FREERATIO)

while True:
    GaussianBlur_ksize = cv2.getTrackbarPos("GaussianBlur_ksize", "TrackBars") * 2 + 1
    GaussianBlur_SigmaX = cv2.getTrackbarPos("GaussianBlur_SigmaX", "TrackBars")
    canny_Th1 = cv2.getTrackbarPos("canny_Th1", "TrackBars")
    canny_Th2 = cv2.getTrackbarPos("canny_Th2", "TrackBars")
    # GaussianBlur3 = cv2.getTrackbarPos("Val Min", "TrackBars")
    # v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    # print(h_min,h_max,s_min,s_max,v_min,v_max)
    # lower = np.array([canny_p1_min, canny_p2_min, v_min])
    # upper = np.array([canny_p1_max, canny_p2_max, v_max])
    # mask = cv2.inRange(imgGray, lower, upper)
    # imgResult = cv2.bitwise_and(img,img,mask=mask)
    imgBlur = cv2.GaussianBlur(imgGray, (GaussianBlur_ksize, GaussianBlur_ksize),
                               GaussianBlur_SigmaX)  # ADD GAUSSIAN BLUR

    imgCanny = cv2.Canny(imgBlur, canny_Th1, canny_Th2)  # APPLY CANNY

    """
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    imgCanny = cv2.Canny(imgBlur, 10, 70)  # APPLY CANNY
    """
    print(GaussianBlur_ksize)
    # cv2.imshow("Original",img)
    # cv2.imshow("HSV",imgHSV)
    # cv2.imshow("Mask", mask)
    # cv2.imshow("Result", imgResult)

    # imgStack = stackImages(0.6, ([img, imgGray], [mask, imgResult]))
    imgStack = stackImages(0.6, ([img, imgBlur, imgCanny]))
    cv2.imshow("Stacked Images", imgStack)

    cv2.waitKey(50)
