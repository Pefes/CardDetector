import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os #read all files in directory
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage import measure



easy = []
medium = []
hard = []
paths = ["images/easy", "images/medium", "images/hard"]
#paths = ["images/easy"]



def readImages():
    for path in paths:
        for r, d, f in os.walk(path):
            for file in f:
                if(path == paths[0]):
                    easy.append(cv.imread(os.path.join(r, file), 1))
                elif(path == paths[1]):
                    medium.append(cv.imread(os.path.join(r, file), 1))
                elif(path == paths[2]):
                    hard.append(cv.imread(os.path.join(r, file), 1))


def showImages(images, number, rows, columns):
    plt.figure(figsize=(20, 20))
    for i in range(1, number + 1):
        plt.subplot(rows, columns, i)
        plt.imshow(images[i - 1], cmap = "gray")
        #plt.imshow(images[i - 1][:, :, ::-1])
    plt.show()
    plt.close()
    plt.clf()


def gamma(image, gamma = 1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv.LUT(image, table)


def bgr2gray(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def contrast(image, perc):
    MIN = np.percentile(image, perc)
    MAX = np.percentile(image, 100-perc)
    image = (image - MIN) / (MAX - MIN)
    image[image[:,:] > 255] = 255
    image[image[:,:] < 0] = 0
    image = image.astype(np.uint8)
    return image


def dilation(image, i):
    kernel = np.ones((5, 5), np.uint8)
    return cv.dilate(image, kernel, iterations = i)


def erosion(image, i):
    kernel = np.ones((5, 5), np.uint8)
    return cv.erode(image, kernel, iterations = i)


def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv.morphologyEx(image, cv.MORPH_OPEN, kernel)


def closing(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)


def laplacian(image):
    return cv.Laplacian(image, cv.CV_64F)


def adaptiveThresh(image):
    #return cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    blur = cv.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return thresh


def thresh(image, value):
    ret, thresh1 = cv.threshold(image, value, 255, cv.THRESH_BINARY)
    return thresh1


def contours(image):
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours


def blur(image, x, y):
    return cv.blur(image, (x, y))


def biBlur(image, x, y, z):
    return cv.bilateralFilter(image, x, y, z)


def clahe(image):
    return claheObj.apply(image)


#def canny(image):
#    return cv.Canny(image, 100, 200)


def gradient(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv.morphologyEx(image, cv.MORPH_GRADIENT, kernel)


def fillHoles(image):#0-1 rgb
    return ndi.binary_fill_holes(image)


def hull(contour):
    return cv.convexHull(contour)


def transformImages(images):
    for i in range(len(images)):
        grayTemp = bgr2gray(images[i])

        #claheTemp = clahe(grayTemp)
        #blurTemp = blur(grayTemp, 5, 5)
        #blurTemp = biBlur(grayTemp, 9, 200, 200)
        #contrastTemp = contrast(blurTemp, 47.5)
        #gammaTemp = gamma(grayTemp, 0.3)

        #claheTemp = clahe(gammaTemp)
        #threshTemp = adaptiveThresh(gammaTemp)
        #erosionTemp = erosion(threshTemp, 5)
        #closingTemp = closing(erosionTemp)

        #contrastTemp = contrast(claheTemp, 25.0)
        #threshTemp = thresh(blurTemp, 150.0)
        #gammaTemp = gamma(claheTemp, 0.7)
        #threshTemp = thresh(gammaTemp, 50.0)


        #images[i] = gamma(images[i], 0.3)
        #images[i] = contrast(images[i], 0.5)
        #images[i] = cv.adaptiveThreshold(images[i], 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 9, 9)
        #images[i] = dilation(images[i], 1)
        #erosionTemp = erosion(threshTemp, 5)
        #dilationTemp = dilation(threshTemp, 1)
        #images[i] = closing(images[i])
        #images[i] = laplacian(images[i])
        #threshTemp = thresh(grayTemp, 127)
        #grayTemp = bgr2gray(images[i])
        contrastTemp = contrast(grayTemp, 48.5)
        gammaTemp = gamma(contrastTemp, 0.55)
        blurTemp = blur(gammaTemp, 15, 15)
        threshTemp = adaptiveThresh(blurTemp)
        erosionTemp = erosion(threshTemp, 4)
        openingTemp = opening(erosionTemp)



        #closingTemp = opening(threshTemp)
        #gradientTemp = gradient(closingTemp)
        #dilationTemp = dilation(gradientTemp, 5)

        #claheTemp = clahe(contrastTemp)
        #gammaTemp = gamma(grayTemp, 5.0)
        #claheTemp = clahe(gammaTemp)
        #threshTemp = adaptiveThresh(claheTemp)
        #threshTemp = thresh(claheTemp, 127)
        #blurTemp = blur(threshTemp, 5, 5)


        contoursTemp = contours(openingTemp)
        contoursTemp3 = contoursTemp
        contoursTemp2 = []

        contoursTemp = [contour for contour in contoursTemp if(cv.contourArea(contour) > 20000 and cv.contourArea(contour) < 250000 and len(cv.approxPolyDP(contour, 0.025 * cv.arcLength(contour, True), True))
 == 4)]
        contoursTemp2 = []

        if(len(contoursTemp) == 0): contoursTemp = contours(openingTemp)

        areas = []
        for contour in contoursTemp:
            areas.append(int(cv.contourArea(contour)))

        if(len(areas) == 0): continue
        medianArea = int(np.median(areas))

        tempAreas = sorted(areas)

        for j in range(len(tempAreas)):
            if(tempAreas[j] > medianArea):
                medianArea = tempAreas[j - 1]
                break

        medianAreaIndex = areas.index(medianArea)
        medianContour = contoursTemp[medianAreaIndex]

        #len(v.approxPolyDP(contour, 0.025 * cv.arcLength(contour, True), True))


        for contour in contoursTemp3:
            #print(cv.matchShapes(contour, medianContour, 1, 0.0))
            if(cv.matchShapes(contour, medianContour, 1, 0.0) < 0.3 and cv.contourArea(contour) > cv.contourArea(medianContour)*0.5):
                contoursTemp2.append(contour)

        '''for contour in contoursTemp:
            peri = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.025 * peri, True)

            #if(cv.contourArea(contour) > 5000 and cv.contourArea(contour) < 300000 and len(approx) == 4):
                #contoursTemp2.append(approx)

            #contoursTemp2.append(hull(contour))
        '''

        cv.drawContours(images[i], contoursTemp2, -1, (0, 255, 0), 20)
        #closingTemp = opening(threshTemp)


        #images[i] = openingTemp
        del grayTemp
        #del blurTemp

    del images


claheObj = cv.createCLAHE(clipLimit = 20.0, tileGridSize=(10, 10)) #creates clahe object
readImages()


transformImages(easy)
transformImages(medium)
transformImages(hard)

showImages(easy, 12, 4, 3)
showImages(medium, 15, 5, 3)
showImages(hard, 16, 4, 4)

del easy
del medium
del hard
cv.destroyAllWindows()
