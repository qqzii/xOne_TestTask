import cv2
from PIL import Image
import pytesseract
import numpy as np


# image = Image.open('docs/1.jpg')
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'




def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#
#
#
# # # Кадрирование
# # cropped = image[10:500, 500:2000]
# # viewImage(cropped, "Кадрирование")
# #
# # # Изменение размера
# # width = int(image.shape[1] * 0.2)
# # height = int(image.shape[0] * 0.2)
# # dim = (width, height)
# # resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
# # viewImage(resized, "Изменение размера")
# #
# # # Поворот
# # (h, w, d) = image.shape
# # center = (w // 2, h // 2)
# # M = cv2.getRotationMatrix2D(center, 180, 1.0)
# # rotated = cv2.warpAffine(image, M, (w, h))
# # viewImage(rotated, "Поворот")
#
#
# Перевод в градации серого и в чёрно-белое изображение по порогу


# blurred = cv2.GaussianBlur(image, (3, 3), 0)
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# viewImage(image, "Поворот")


image = cv2.imread("docs/1.jpg")
viewImage(image, "1")
width = int(image.shape[1] * 4)
height = int(image.shape[0] * 4)
dim = (width, height)
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
viewImage(resized, "1")
# cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#
# cv2.threshold(cv2.bilateralFilter(img, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#
# cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#
# cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
#
# cv2.adaptiveThreshold(cv2.bilateralFilter(img, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

# cv2.adaptiveThreshold(cv2.medianBlur(image, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)



# config = r'--oem 3 --psm 13'
# text = pytesseract.image_to_string(image, lang='rus', config=config)
# print(text)
