import cv2


camR = cv2.VideoCapture(1)

while True:
    
    retR, frameR = camR.read()
    frameR = cv2.flip(frameR, -1)
    h, w, _ = frameR.shape

    left = frameR[:, :w//2]
    right = frameR[:, w//2:]

    cv2.imshow("Left", left)
    cv2.imshow("Right", right)
    # cv2.imshow("Right", frameR)

    if cv2.waitKey(1) == 27:
        break