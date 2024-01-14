import cv2
import numpy as np

if __name__ == "__main__":
    image = cv2.imread("233.jpg")
    origin = image.copy()
    print(origin.shape)
    (B, G, R) = cv2.split(origin)
    zeros = np.zeros(image.shape[:2], dtype = "uint8")

    imgR = cv2.merge([zeros, zeros, R])

    imgG = cv2.merge([zeros, G, zeros])

    imgB = cv2.merge([B, zeros, zeros])

    cv2.namedWindow("R",0)
    cv2.imshow("R",imgR)
    cv2.namedWindow("B",0)
    cv2.imshow("B",imgB)

    new = cv2.merge([B,R])

    BR = cv2.merge([B, zeros, R])

    cv2.namedWindow("new",0)
    cv2.imshow("new",BR)
    cv2.waitKey(0)
    print(new.shape)
    # image = new
    image = new.transpose((2,0,1)[::-1])
    print(image.shape)
