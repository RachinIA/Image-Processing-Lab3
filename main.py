import cv2
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import tkinter as tk
from scipy import ndimage
from scipy.ndimage.filters import convolve
from numba import jit

def fileOpen():
    ###Choose image on local machin###
    fIn=tk.filedialog.askopenfilename(title="Select image", filetypes=[("Image",["jpg", "*.jpg", "*.png", "*.jpeg"])])
    if(fIn):  ##Success
        return fIn
    else:  ##Fail
        return -1
    
### Creation of Gaussian kernel### 
def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    return g
#############################

###Gradient searching ###
def sobel_filter(image):
    xK = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    yK = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    x = np.abs(convolve(image, xK))
    y = np.abs(convolve(image, yK))
    x = x * 255 / np.max(x)
    y = y * 255 / np.max(y)

    xy = np.sqrt((np.square(x)) + (np.square(y)))
    result = xy * 255 / np.max(xy)
    theta = np.arctan2(y, x)
    return result, theta
######################

### Non-Maximum Suppression ###
def nonMaximal(img, d):
    m, n = img.shape
    z = np.zeros((m, n), dtype=np.int32)
    angle = d * 180. / np.pi
    angle[angle < 0] += 180
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            q = 255
            r = 255
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = img[i, j + 1]
                r = img[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = img[i + 1, j - 1]
                r = img[i - 1, j + 1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = img[i + 1, j]
                r = img[i - 1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = img[i - 1, j - 1]
                r = img[i + 1, j + 1]
            if (img[i, j] >= q) and (img[i, j] >= r):
                z[i, j] = img[i, j]
            else:
                z[i, j] = 0
    return z
##############################

### Double threshold filtering ###
def DTF(img, weak_pixel, strong_pixel, low, high):
    high = img.max() * high
    low = high * low
    m, n = img.shape
    res = np.zeros((m, n), dtype=np.int32)
    weak = np.int32(weak_pixel)
    strong = np.int32(strong_pixel)
    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img <= high) & (img >= low))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return res
##########################

### Tracing ###
def tracing(img, weak_pixel, strong_pixel):
    m, n = img.shape
    weak = weak_pixel
    strong = strong_pixel
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            if img[i, j] == weak:
                if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                        or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                        or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (
                                img[i - 1, j + 1] == strong)):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img
##############

### Canny's Algorithm ###
def cannyAlg(imageOrig):
    new_image = imageOrig
    """ Gray filter from first Lab """
    Red = new_image[:,:,0]
    Green = new_image[:,:,1]
    Blue = new_image[:,:,2]
    average = (Blue*0.148 + Green*0.5547 + Red*0.2952) 
    new_image = average
    """"""""""""""""""""""""""""""
    """ Gaussian  filter """
    new_image = convolve(new_image, gaussian_kernel(5, 1))
    """"""""""""""""""""""""""""""""""""
    """ Gradient searching """
    new_image, theta = sobel_filter(new_image)
    """"""""""""""""""""""""""""""
    """ Non-Maximum Suppression """
    new_image = nonMaximal(new_image, theta)
    """"""""""""""""""""""""""""""""""""
    """ Double threshold filtering """
    new_image = DTF(new_image, 75, 255, 0.05, 0.15)
    """"""""""""""""""""""""""""""""""""
    """ Tracing """
    new_image = tracing(new_image, 75, 255)
    """"""""""""""""""
    return  new_image
#######################

### Visual Canny test ###
def visualTestCanny():
        file = fileOpen()
        if (file == -1):
            sys.exit()
        image = cv2.imread(file)
        cv2.imshow("Image", image)
        canny = cannyAlg(image)
        cv2.imshow("Canny's Algorithm",  np.uint8(canny))
        cv2.waitKey(0)  #Uncomment to wait key befor closing image
        cv2.destroyAllWindows()  #Uncomment to close all the windows
##############
        
""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""
"""      Hough transform      """
""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""

### Gaussian ###
def gaussianSmoothing(input_img):                         
    gaussian_filter=np.array([[0.109,0.111,0.109],
                                              [0.111,0.135,0.111],
                                              [0.109,0.111,0.109]])
    return cv2.filter2D(input_img,-1,gaussian_filter)
##############

### Сanny ###
def cannyDetection(input):    
    input = input.astype('uint8')
    thres, ret_matrix = cv2.threshold(input,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)    
    lower_threshold = thres * 0.4
    upper_threshold = thres * 1.3
    edges = cv2.Canny(input, lower_threshold, upper_threshold)
    return edges
###############

### Haff's circles ###
@jit(nopython=True, parallel=True)
def HoughCircles(input,circles): 
    rows = input.shape[0] 
    cols = input.shape[1]
    length=int(rows/2)
    radius = [i for i in range(5,length)]
    threshold = 190 
    
    for r in radius:
        acc_cells = np.full((rows,cols),fill_value=0,dtype=np.uint64)
        for x in range(rows): 
            for y in range(cols): 
                if input[x][y] == 255: 
                    for angle in range(0,360): 
                        b = y - round(r * np.sin(angle * np.pi / 180))
                        a = x - round(r * np.cos(angle * np.pi / 180))
                        if a >= 0 and a < rows and b >= 0 and b < cols: 
                            acc_cells[a][b] += 1                            
        acc_cell_max = np.amax(acc_cells)
        if(acc_cell_max > 150):  
            for x in range(rows):
                for y in range(cols):
                    if acc_cells[x][y] < 150:
                        acc_cells[x][y] = 0
            for i in range(rows): 
                for j in range(cols): 
                    if(i > 0 and j > 0 and i < rows-1 and j < cols-1 and acc_cells[i][j] >= 150):
                        avg_sum = np.float32((acc_cells[i][j]+acc_cells[i-1][j]+acc_cells[i+1][j]+acc_cells[i][j-1]+acc_cells[i][j+1]+acc_cells[i-1][j-1]+acc_cells[i-1][j+1]+acc_cells[i+1][j-1]+acc_cells[i+1][j+1])/9) 
                        if(avg_sum >= 33):
                            circles.append((i,j,r))
                            acc_cells[i:i+5,j:j+7] = 0
###############

### Analysis ###
@jit(nopython=True, parallel=True)
def analyze(O, image):
    rows = image.shape[0]
    clmns = image.shape[1]
    number = 0
    for c in O:
        number += 1
        x = c[0]
        y = c[1]
        R = c[2]
        S = 0
        P = 0
        for i in range(rows):
            for j in range(clmns):
                if (i - x)**2 + (j - y)**2 <= R*R:
                    S += 1
        for i in range(rows):
            for j in range(clmns):
                if (i - x)**2 + (j - y)**2 <= R*R:
                    if ((i + 1 - x)**2 + (j - y)**2 > R*R or (i - 1 - x)**2 + (j - y)**2 > R*R or (i - x)**2 + (j + 1 - y)**2 > R*R or (i - x)**2 + (j - 1 - y)**2 > R*R):
                        P += 1
        print("Circle number ", number, ' has coordinates = (', x,', ', y,'), radius = ', R,', perimeter = ', P,', square = ', S)
##################

### Visual Haff test ###
def visualTestHaff():
        file = fileOpen()
        if (file == -1):
            sys.exit()
        image = cv2.imread(file)
        cv2.imshow("Image", image)
        new_image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        """ Gaussian  """
        new_image = gaussianSmoothing(new_image)
        """"""""""""""""""""""""""""""
        """ Canny """
        new_image = cannyDetection(new_image)
        """"""""""""""""""
        circles = [(0, 0, 0)]
        
        HoughCircles(new_image, circles)
        
        """ Analysis """
        analyze(circles, image)
        """"""""""""""""""
        for vertex in circles:
            cv2.circle(image, (vertex[1],vertex[0]), vertex[2], (127, 255, 0), 1)
        cv2.imshow('Image with detected circles', image) 
        cv2.waitKey(0)  #Uncomment to wait key befor closing image
        cv2.destroyAllWindows()  #Uncomment to close all the windows
##############

### Comparison in time ###
def compareMineVsCV():
    file = fileOpen()
    if (file == -1):
        sys.exit()
    image = cv2.imread(file)
    ################################
    """Create window with results"""
    windowResults = tk.Toplevel(root)
    lbMy=tk.Label(windowResults, text="||\n||\n||\n||\n||").grid(row = 0, column = 1,rowspan = 3)
    lbMy=tk.Label(windowResults, text="Time").grid(row = 0, column = 2)
    lbMy=tk.Label(windowResults, text="||\n||\n||\n||\n||").grid(row = 0, column = 3,rowspan = 3)
    lbMy=tk.Label(windowResults, text="My Canny realization").grid(row = 1, column = 0)
    lbMy=tk.Label(windowResults, text="CV Canny realization").grid(row = 2, column = 0)
    ############################
    ### CV Canny ###
    startTime = time.time()
    cvCanny = cv2.Canny(image, 75, 255)
    cvResultTime= time.time() - startTime
    cv2.imshow('CV Canny Image', cvCanny) 
    ###############
    ### My Canny ###
    startTime = time.time()
    myCanny = cannyAlg(image)
    myResultTimeCanny = time.time() - startTime
    cv2.imshow('My Canny Image', np.uint8(myCanny)) 
    #############
    ################################
    """Change window with results"""
    lbMy=tk.Label(windowResults, text=str(round(myResultTimeCanny,4))).grid(row = 1, column = 2)
    lbMy=tk.Label(windowResults, text=str(round(cvResultTime,4))).grid(row = 2, column = 2)
    ################################

######################
"""Buttons and form"""
######################
root = tk.Tk()
hsvBut1 = tk.Button(root, text = 'Visual test of Canny\'s algorithm', activebackground = "#555555", command = visualTestCanny).grid(row = 0, column = 0)
hsvBut1 = tk.Button(root, text = 'Haff\'s method and analysis', activebackground = "#555555", command = visualTestHaff).grid(row = 0, column = 2)
hsvBut2 = tk.Button(root, text = 'Comparison of my and cv realization', activebackground = "#555555", command = compareMineVsCV).grid(row = 0, column = 4)
lbFreeSpace = tk.Label(root, text = '||').grid(row = 0, column = 1)
lbFreeSpace = tk.Label(root, text = '||').grid(row = 0, column = 3)
root.mainloop()
######################
