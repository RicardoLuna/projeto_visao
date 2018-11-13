import cv2
import numpy as np
from matplotlib import pyplot as plt

def determining_iris_region(img):
    rgb = img.copy()
    img[:,:,0] = 0
    img[:,:,1] = 0
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Z = gray_image.reshape((-1,1))
    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    _,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((gray_image.shape))
    min_value = res2.min()
    rows,cols = res2.shape
    # Aplica um destaque para região do olho criando um imagem binarizada
    for i in range(rows):
        for j in range(cols):
            if res2[i,j] == min_value:
                res2[i,j] = 255
            else:
                res2[i,j] = 0

    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(res2, cv2.MORPH_CLOSE, kernel)
    _, contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    area = []
    #calcula a area de cada elemeto
    for i in range(1,len(contours)):
        area.append(cv2.contourArea(contours[i]))

    max_pos = area.index(max(area))
    x, y, width, height = cv2.boundingRect(contours[max_pos+1])
    # Aplica o recorte para a posição x até o x + largura e y até o y + altura
    crop = rgb[y:y+height, x:x+width]
    return crop


def rgb2ycbcr(img):
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    y, _, _ = cv2.split(img2)
    return y

def auto_canny(image, sigma=0.33):
    lowThreshold = 80
    ratio = 3
    kernel_size = 3
    new_img = rgb2ycbcr(image)
	# compute the median of the single channel pixel intensities
    #ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# apply automatic Canny edge detection using the computed median
    edged = cv2.Canny(img, 313, 14)
 
	# return the edged image
    return edged

def nothing(x):
    pass

img = cv2.imread('iris.jpg')
crop = determining_iris_region(img)

 
img2 = cv2.cvtColor(crop, cv2.COLOR_BGR2YCR_CB)
y, _, _ = cv2.split(img2)
img = cv2.blur(y, (5,5))
 
canny_edge = cv2.Canny(img, 0, 0)
 
cv2.imshow('image', img)
cv2.imshow('canny_edge', canny_edge)
 
cv2.createTrackbar('min_value','canny_edge',0,500,nothing)
cv2.createTrackbar('max_value','canny_edge',0,500,nothing)
 
while(1):
    cv2.imshow('image', img)
    cv2.imshow('canny_edge', canny_edge)
     
    min_value = cv2.getTrackbarPos('min_value', 'canny_edge')
    max_value = cv2.getTrackbarPos('max_value', 'canny_edge')
 
    canny_edge = cv2.Canny(img, min_value, max_value)
     
    k = cv2.waitKey(37)
    if k == 27:
        break
# canny = auto_canny(crop)
# cv2.imshow('result', canny)
# cv2.waitKey(0)
# cv2.destroyAllWindows()