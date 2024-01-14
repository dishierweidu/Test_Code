import cv2
from cv2 import dnn

src = cv2.imread("233.jpg")
# roi = src[2469:548, 2707:775]
roi = src[548:775, 2469:2707 ]

cv2.imshow('roi',roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
