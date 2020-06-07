from pytesseract import pytesseract as tesseract
import numpy as np
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import sys
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from scipy.integrate import quad

def load_image(path):
    """
    Loads image using Pillow library as grayscale image and return image as numpy array
    Input: image path as a string
    Returns: grayscale image
    """
    img=cv2.imread(path,cv2.IMREAD_UNCHANGED)

    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def remove_noise(image):
    """
    Uses cv2 medianBlur to remove noise
    """
    return cv2.medianBlur(image,ksize=3)

def canny(image):
    can=cv2.Canny(image,100,200,apertureSize=5,L2gradient=True)
    # plt.imshow(can,cmap="gray")
    # plt.waitforbuttonpress(0)
    # plt.close()
    return can


def sobel(image):
    sobel2Dy=np.array([
        [-1,-2,-1],
        [0,0,0],
        [1,2,1]
    ],dtype=np.float64)

    sobel2Dx=sobel2Dy.T

    nx=np.zeros(shape=image.shape,dtype=np.int32)
    ny = np.zeros(shape=image.shape, dtype=np.int32)
    grad_mag=np.zeros(shape=image.shape, dtype=np.int32)
    grad_theta = np.zeros(shape=image.shape, dtype=np.float32)

    for r in range(image.shape[0]):
        ith = r - 1
        for c in range(image.shape[1]):
            jth = c - 1
            # If kernel exceeds image size, make pixel 0 at that location
            if ith < 0 or jth < 0:
                nx[r, c] = 0
                ny[r, c] = 0
                grad_mag[r,c]=0
            elif ith + 3 > img.shape[0] or jth + 3 > image.shape[1]:
                nx[r, c] = 0
                ny[r, c] = 0
                grad_mag[r,c]=0
            else:
                # Convolution operation
                nx[r, c] = round((image[ith:ith + 3, jth:jth + 3] * sobel2Dx).sum(), 0)
                ny[r, c] = round((image[ith:ith + 3, jth:jth + 3] * sobel2Dy).sum(), 0)
                grad_mag[r,c] = round(np.sqrt(nx[r,c]*nx[r,c]+ny[r,c]*ny[r,c]),0)
                if nx[r,c]!=0:
                    grad_theta[r, c] = np.arctan(ny[r,c]/nx[r,c])
                else:
                    grad_theta[r,c] = np.inf

    # plt.imshow(grad_mag,cmap="gray")
    # plt.waitforbuttonpress(0)
    # plt.close()

    # print(grad_theta,grad_theta.shape)
    grad_mag=grad_mag/np.max(grad_mag)*255

    return grad_mag,grad_theta

def integrand(a,b,c,x):
    return a*x**2+b*x+c

def quadratic_fit(data):
    """
    Input: indices of values not equal to 0

    Outputs (3): coefficients, Root mean square error,
    and R_squared
    """

    x=data[:,1].reshape(-1,1)
    y=data[:,0].reshape(-1,1)
    polynomial_features = PolynomialFeatures(degree=2)
    x_poly = polynomial_features.fit_transform(x)
    model = LinearRegression(n_jobs=2)
    reg = model.fit(x_poly, y)

    y_poly_pred = model.predict(x_poly)

    rmse = np.sqrt(mean_squared_error(y, y_poly_pred))
    r2 = r2_score(y, y_poly_pred)
    coef=reg.coef_

    # plt.scatter(x, y, s=10)
    # plt.scatter(x,y_poly_pred,marker="d")
    # plt.show()

    return coef,rmse,r2

def water_fill(image):

    #remove blackspaces from top,bottom,left,right of image
    indices=np.argwhere(image!=0)
    topRow,leftColumn=np.min(indices,axis=0)
    bottomRow,rightColumn=np.max(indices,axis=0)
    image=image[topRow:bottomRow+1,leftColumn:rightColumn+1]

    h, w = image.shape

    #Calculate up jar capacity
    #Find the row closest to row 0 that !=0 in the middle of image
    rw=np.min(np.argwhere(image[:,w//2-1:w//2+2]!=0)[:,0])
    partial_img=image[:rw+1,:]
    indices = np.argwhere(partial_img != 0)  # row,column
    indices[:, 0] = rw + 1 - indices[:, 0]  # so image is outputted upright
    coef,rmse,rSq=quadratic_fit(indices)
    lb=np.min(indices[:,1]) #lower bound to integrate over
    ub=np.max(indices[:,1]) #upper bound to integrate over
    Iup = quad(integrand, a=lb, b=ub,
               args=(coef[0,2], coef[0,1], coef[0,0]))

    #Calculate down jar capacity
    #Find the row closest to row 0 that !=0 in the middle of image
    rw=np.max(np.argwhere(image[:,w//2-1:w//2+2]!=0))
    partial_img=image[rw:,:]
    indices = np.argwhere(partial_img != 0)  # row,column
    indices[:, 0] = h - indices[:, 0]  # so image is outputted upright
    coef,rmse,rSq=quadratic_fit(indices)
    lb=np.min(indices[:,1]) #lower bound to integrate over
    ub=np.max(indices[:,1]) #upper bound to integrate over
    Idown = quad(integrand, a=lb, b=ub,
               args=(coef[0,2], coef[0,1], coef[0,0]))

    return abs(Iup[0]),abs(Idown[0])

def water_fill25(image):

    #remove blackspaces from top,bottom,left,right of image
    indices=np.argwhere(image!=0)
    topRow,leftColumn=np.min(indices,axis=0)
    bottomRow,rightColumn=np.max(indices,axis=0)
    image=image[topRow:bottomRow+1,leftColumn:rightColumn+1]

    h, w = image.shape

    #Calculate up jar capacity
    #Find the row closest to row 0 that !=0 in the middle of image
    rw=np.min(np.argwhere(image[:,w//2-1:w//2+2]!=0)[:,0])
    partial_img=image[:rw+1,:]
    indices = np.argwhere(partial_img != 0)  # row,column
    indices[:, 0] = rw + 1 - indices[:, 0]  # so image is outputted upright
    coef,rmse,rSq=quadratic_fit(indices)
    lb=np.min(indices[:,1]) #lower bound to integrate over
    ub=np.max(indices[:,1]) #upper bound to integrate over
    Iup = quad(integrand, a=lb, b=ub,
               args=(coef[0,2], coef[0,1], coef[0,0]))

    #Calculate down jar capacity
    #Find the row closest to row 0 that !=0 in the middle of image
    rw=np.max(np.argwhere(image[:,w//2-1:w//2+2]!=0)[:,0])
    partial_img=image[rw:,:]
    indices = np.argwhere(partial_img != 0)  # row,column
    indices[:, 0] = h - indices[:, 0]  # so image is outputted upright
    coef,rmse,rSq=quadratic_fit(indices)
    lb=np.min(indices[:,1]) #lower bound to integrate over
    ub=np.max(indices[:,1]) #upper bound to integrate over
    Idown = quad(integrand, a=lb, b=ub,
               args=(coef[0,2], coef[0,1], coef[0,0]))

    #Calculate right jar capacity
    # Find column furthest right that !=0 in middle of image
    col=np.max(np.argwhere(image[h//2-1:h//2+2,:]!=0)[:,1])
    partial_img=image[:,col:]
    indices = np.argwhere(partial_img != 0)  # row,column
    indices[:, 0] = h - indices[:, 0]  # so image is outputted upright
    indices=np.flip(indices,axis=1) #swap row and column indices
    coef,rmse,rSq=quadratic_fit(indices)
    lb=np.min(indices[:,1]) #lower bound to integrate over
    ub=np.max(indices[:,1]) #upper bound to integrate over
    Iright = quad(integrand, a=lb, b=ub,
               args=(coef[0,2], coef[0,1], coef[0,0]))

    #Calculate left jar capacity
    # Find column furthest right that !=0 in middle of image
    col=np.min(np.argwhere(image[h//2-1:h//2+2,:]!=0)[:,1])
    partial_img=image[:,:col+1]
    indices = np.argwhere(partial_img != 0)  # row,column
    indices[:, 0] = h - indices[:, 0]  # so image is outputted upright
    indices=np.flip(indices,axis=1) #swap row and column indices
    coef,rmse,rSq=quadratic_fit(indices)
    lb=np.min(indices[:,1]) #lower bound to integrate over
    ub=np.max(indices[:,1]) #upper bound to integrate over
    Ileft = quad(integrand, a=lb, b=ub,
               args=(coef[0,2], coef[0,1], coef[0,0]))

    return abs(Iup[0]),abs(Idown[0]),abs(Ileft[0]),abs(Iright[0])

def water_fill2(image):

    #remove blackspaces from top,bottom,left,right of image
    indices=np.argwhere(image!=0)
    topRow,leftColumn=np.min(indices,axis=0)
    bottomRow,rightColumn=np.max(indices,axis=0)
    image=image[topRow:bottomRow+1,leftColumn:rightColumn+1]

    h, w = image.shape

    #Calculate up jar capacity
    #Find the row closest to 0 in a column that !=0
    rw=np.min(np.argwhere(image[:,w//2]!=0)[:,0])
    col=w//2
    if np.min(np.argwhere(image[:,w//3]!=0)[:,0])>rw:
        rw=np.min(np.argwhere(image[:,w//3]!=0)[:,0])
        col=w//3
    if np.min(np.argwhere(image[:,2*w//3]!=0)[:,0])>rw:
        rw = np.min(np.argwhere(image[:, 2*w // 3] != 0)[:, 0])
        col=2*w//3
    i=rw
    Iup = 0
    #while the letter is concave, increase up jar capacity
    while i>-1:
        if np.any(image[i,:col]) and np.any(image[i,col+1:]):
            i-=1
            Iup+=1
        else:
            break

    #Calculate down jar capacity
    #Find the row furthest down in a column that !=0
    rw=np.max(np.argwhere(image[:,w//2]!=0)[:,0])
    col=w//2
    if np.max(np.argwhere(image[:,w//3]!=0)[:,0])<rw:
        rw=np.max(np.argwhere(image[:,w//3]!=0)[:,0])
        col=w//3
    if np.max(np.argwhere(image[:,2*w//3]!=0)[:,0])<rw:
        rw = np.max(np.argwhere(image[:, 2*w // 3] != 0)[:, 0])
        col=2*w//3
    i=rw
    Idown=0
    while i<h:
        if np.any(image[i, :col]) and np.any(image[i, col + 1:]):
            i+=1
            Idown+=1
        else:
            break

    if abs(Iup-Idown)<=1:
        #Calculate left jar capacity
        col=np.max(np.argwhere(image[h//2,:]!=0)[:,0]) #rightmost col
        rw=h//2
        if np.max(np.argwhere(image[h//3,:]!=0)[:,0])<col:
            col=np.max(np.argwhere(image[h//3,:]!=0)[:,0])
            rw=h//3
        if np.max(np.argwhere(image[2*h//3,:]!=0)[:,0])<col:
            col = np.max(np.argwhere(image[2*h//3, :] != 0)[:, 0])
            rw=2*h//3
        i=col
        Ileft=0
        while i>-1:
            if np.any(image[:rw, i]) and np.any(image[rw+1:, i]):
                i-=1
                Ileft+=1
            else:
                break

        #Calculate right jar capacity
        col=np.min(np.argwhere(image[h//2,:]!=0)[:,0]) #leftmost col
        rw=h//2
        if np.min(np.argwhere(image[h//3,:]!=0)[:,0])>col:
            col=np.min(np.argwhere(image[h//3,:]!=0)[:,0])
            rw=h//3
        if np.min(np.argwhere(image[2*h//3,:]!=0)[:,0])>col:
            col = np.max(np.argwhere(image[2*h//3, :] != 0)[:, 0])
            rw=2*h//3
        i=col
        Iright=0
        while i<w:
            if np.any(image[:rw, i]) and np.any(image[rw+1:, i]):
                i+=1
                Iright+=1
            else:
                break
    else:
        Ileft=Iright=0

    return Iup,Idown,Ileft,Iright

def water_fill5(image):

    #remove blackspaces from top,bottom,left,right of image
    indices=np.argwhere(image!=0)
    topRow,leftColumn=np.min(indices,axis=0)
    bottomRow,rightColumn=np.max(indices,axis=0)
    image=image[topRow:bottomRow+1,leftColumn:rightColumn+1]

    h, w = image.shape

    #Calculate up jar capacity
    #Find the row closest to row 0 that !=0 in the middle of image
    rw=np.min(np.argwhere(image[:,w//2-1:w//2+2]!=0)[:,0])
    partial_img=image[:rw+1,:]
    indices = np.argwhere(partial_img != 0)  # row,column
    indices[:, 0] = rw + 1 - indices[:, 0]  # so image is outputted upright
    coef,rmse,rSq=quadratic_fit(indices)
    lb=np.min(indices[:,1]) #lower bound to integrate over
    ub=np.max(indices[:,1]) #upper bound to integrate over
    Iup = quad(integrand, a=lb, b=ub,
               args=(coef[0,2], coef[0,1], coef[0,0]))
    y1=abs(coef[0, 0] * lb ** 2 + coef[0, 1] * lb + coef[0, 2])
    y2=abs(coef[0, 0] * ub ** 2 + coef[0, 1] * ub + coef[0, 2])
    if coef[0,2]<0:
        Iup=abs(Iup[0])
    else:
        rectHeight = max(y1,y2)
        Iup=rectHeight*abs(lb-ub)-abs(Iup[0])

    #Calculate down jar capacity
    #Find the row closest to row 0 that !=0 in the middle of image
    rw=np.max(np.argwhere(image[:,w//2-1:w//2+2]!=0)[:,0])
    partial_img=image[rw:,:]
    indices = np.argwhere(partial_img != 0)  # row,column
    indices[:, 0] = h - indices[:, 0]  # so image is outputted upright
    coef,rmse,rSq=quadratic_fit(indices)
    lb=np.min(indices[:,1]) #lower bound to integrate over
    ub=np.max(indices[:,1]) #upper bound to integrate over
    Idown = quad(integrand, a=lb, b=ub,
               args=(coef[0,2], coef[0,1], coef[0,0]))
    y1=abs(coef[0, 0] * lb ** 2 + coef[0, 1] * lb + coef[0, 2])
    y2=abs(coef[0, 0] * ub ** 2 + coef[0, 1] * ub + coef[0, 2])
    if coef[0,2]<0:
        Idown=abs(Idown[0])
    else:
        rectHeight = max(y1,y2)
        Idown=rectHeight*abs(lb-ub)-abs(Idown[0])

    #Calculate right jar capacity
    # Find column furthest right that !=0 in middle of image
    col=np.max(np.argwhere(image[h//2-1:h//2+2,:]!=0)[:,1])
    partial_img=image[:,col:]
    indices = np.argwhere(partial_img != 0)  # row,column
    indices[:, 0] = h - indices[:, 0]  # so image is outputted upright
    indices=np.flip(indices,axis=1) #swap row and column indices
    coef,rmse,rSq=quadratic_fit(indices)
    lb=np.min(indices[:,1]) #lower bound to integrate over
    ub=np.max(indices[:,1]) #upper bound to integrate over
    Iright = quad(integrand, a=lb, b=ub,
               args=(coef[0,2], coef[0,1], coef[0,0]))
    y1=abs(coef[0, 0] * lb ** 2 + coef[0, 1] * lb + coef[0, 2])
    y2=abs(coef[0, 0] * ub ** 2 + coef[0, 1] * ub + coef[0, 2])
    if coef[0,2]<0:
        Iright=abs(Iright[0])
    else:
        rectHeight = max(y1,y2)
        Iright=rectHeight*abs(lb-ub)-abs(Iright[0])

    #Calculate left jar capacity
    # Find column furthest right that !=0 in middle of image
    col=np.min(np.argwhere(image[h//2-1:h//2+2,:]!=0)[:,1])
    partial_img=image[:,:col+1]
    indices = np.argwhere(partial_img != 0)  # row,column
    indices[:, 0] = h - indices[:, 0]  # so image is outputted upright
    indices=np.flip(indices,axis=1) #swap row and column indices
    coef,rmse,rSq=quadratic_fit(indices)
    lb=np.min(indices[:,1]) #lower bound to integrate over
    ub=np.max(indices[:,1]) #upper bound to integrate over
    Ileft = quad(integrand, a=lb, b=ub,
               args=(coef[0,2], coef[0,1], coef[0,0]))
    y1=abs(coef[0, 0] * lb ** 2 + coef[0, 1] * lb + coef[0, 2])
    y2=abs(coef[0, 0] * ub ** 2 + coef[0, 1] * ub + coef[0, 2])
    if coef[0,2]<0:
        Ileft=abs(Ileft[0])
    else:
        rectHeight = max(y1,y2)
        Ileft=rectHeight*abs(lb-ub)-abs(Ileft[0])

    return Iup,Idown,Ileft,Iright,h,w

def water_fill6(image):

    #remove blackspaces from top,bottom,left,right of image
    indices=np.argwhere(image!=0)
    topRow,leftColumn=np.min(indices,axis=0)
    bottomRow,rightColumn=np.max(indices,axis=0)
    image=image[topRow:bottomRow+1,leftColumn:rightColumn+1]

    h, w = image.shape

    #Calculate up jar capacity
    #Find the row closest to row 0 that !=0 in the middle of image
    rw=np.min(np.argwhere(image[:,w//2]!=0)[:,0])
    col=w//2
    partial_img = image[:rw + 1, :]
    if np.min(np.argwhere(image[:,w//3]!=0)[:,0])>rw:
        rw=np.min(np.argwhere(image[:,w//3]!=0)[:,0])
        col=w//3
        partial_img = image[:rw + 1, :col+1]
    if np.min(np.argwhere(image[:,2*w//3]!=0)[:,0])>rw:
        rw = np.min(np.argwhere(image[:, 2*w // 3] != 0)[:, 0])
        col=2*w//3
        partial_img = image[:rw + 1, col:]
    indices = np.argwhere(partial_img != 0)  # row,column
    indices[:, 0] = rw + 1 - indices[:, 0]  # so image is outputted upright
    coef,rmse,rSq=quadratic_fit(indices)
    lb=np.min(indices[:,1]) #lower bound to integrate over
    ub=np.max(indices[:,1]) #upper bound to integrate over
    Iup = quad(integrand, a=lb, b=ub,
               args=(coef[0,2], coef[0,1], coef[0,0]))
    y1=abs(coef[0, 0] * lb ** 2 + coef[0, 1] * lb + coef[0, 2])
    y2=abs(coef[0, 0] * ub ** 2 + coef[0, 1] * ub + coef[0, 2])
    if coef[0,2]<0:
        Iup=abs(Iup[0])
    else:
        rectHeight = max(y1,y2)
        Iup=rectHeight*abs(lb-ub)-abs(Iup[0])

    #Calculate down jar capacity
    #Find the row closest to row 0 that !=0 in the middle of image
    rw=np.min(np.argwhere(image[:,w//2]!=0)[:,0])
    col=w//2
    partial_img=image[rw:,:]
    if np.min(np.argwhere(image[:,w//3]!=0)[:,0])<rw:
        rw=np.min(np.argwhere(image[:,w//3]!=0)[:,0])
        col=w//3
        partial_img = image[rw:,:col+1]
    if np.min(np.argwhere(image[:,2*w//3]!=0)[:,0])<rw:
        rw = np.min(np.argwhere(image[:, 2*w // 3] != 0)[:, 0])
        col=2*w//3
        partial_img = image[rw:, col:]
    indices = np.argwhere(partial_img != 0)  # row,column
    indices[:, 0] = h - indices[:, 0]  # so image is outputted upright
    coef,rmse,rSq=quadratic_fit(indices)
    lb=np.min(indices[:,1]) #lower bound to integrate over
    ub=np.max(indices[:,1]) #upper bound to integrate over
    Idown = quad(integrand, a=lb, b=ub,
               args=(coef[0,2], coef[0,1], coef[0,0]))
    y1=abs(coef[0, 0] * lb ** 2 + coef[0, 1] * lb + coef[0, 2])
    y2=abs(coef[0, 0] * ub ** 2 + coef[0, 1] * ub + coef[0, 2])
    if coef[0,2]<0:
        Idown=abs(Idown[0])
    else:
        rectHeight = max(y1,y2)
        Idown=rectHeight*abs(lb-ub)-abs(Idown[0])

    #Calculate right jar capacity
    # Find column furthest right that !=0 in middle of image
    col=np.max(np.argwhere(image[h//2,:]!=0)[:,0])
    rw=h//2
    partial_img=image[:,col:]
    if np.max(np.argwhere(image[h//3,:]!=0)[:,0])<col:
        col=np.max(np.argwhere(image[h//3,:]!=0)[:,0])
        rw=h//3
        partial_img = image[:rw+1,col:]
    if np.max(np.argwhere(image[2*h//3,:]!=0)[:,0])<col:
        col = np.max(np.argwhere(image[2*h//3, :] != 0)[:, 0])
        rw=2*h//3
        partial_img = image[rw:, col:]
    indices = np.argwhere(partial_img != 0)  # row,column
    indices[:, 0] = h - indices[:, 0]  # so image is outputted upright
    indices=np.flip(indices,axis=1) #swap row and column indices
    coef,rmse,rSq=quadratic_fit(indices)
    lb=np.min(indices[:,1]) #lower bound to integrate over
    ub=np.max(indices[:,1]) #upper bound to integrate over
    Iright = quad(integrand, a=lb, b=ub,
               args=(coef[0,2], coef[0,1], coef[0,0]))
    y1=abs(coef[0, 0] * lb ** 2 + coef[0, 1] * lb + coef[0, 2])
    y2=abs(coef[0, 0] * ub ** 2 + coef[0, 1] * ub + coef[0, 2])
    if coef[0,2]<0:
        Iright=abs(Iright[0])
    else:
        rectHeight = max(y1,y2)
        Iright=rectHeight*abs(lb-ub)-abs(Iright[0])

    #Calculate left jar capacity
    # Find column furthest right that !=0 in middle of image
    col=np.min(np.argwhere(image[h//2,:]!=0)[:,0])
    rw = h // 2
    partial_img=image[:,:col+1]
    if np.min(np.argwhere(image[h//3,:]!=0)[:,0])>col:
        col=np.min(np.argwhere(image[h//3,:]!=0)[:,0])
        rw=h//3
        partial_img = image[:rw+1,:col+1]
    if np.min(np.argwhere(image[2*h//3,:]!=0)[:,0])>col:
        col = np.min(np.argwhere(image[2*h//3, :] != 0)[:, 0])
        rw=2*h//3
        partial_img = image[rw:, :col+1]
    indices = np.argwhere(partial_img != 0)  # row,column
    indices[:, 0] = h - indices[:, 0]  # so image is outputted upright
    indices=np.flip(indices,axis=1) #swap row and column indices
    coef,rmse,rSq=quadratic_fit(indices)
    lb=np.min(indices[:,1]) #lower bound to integrate over
    ub=np.max(indices[:,1]) #upper bound to integrate over
    Ileft = quad(integrand, a=lb, b=ub,
               args=(coef[0,2], coef[0,1], coef[0,0]))
    y1=abs(coef[0, 0] * lb ** 2 + coef[0, 1] * lb + coef[0, 2])
    y2=abs(coef[0, 0] * ub ** 2 + coef[0, 1] * ub + coef[0, 2])
    if coef[0,2]<0:
        Ileft=abs(Ileft[0])
    else:
        rectHeight = max(y1,y2)
        Ileft=rectHeight*abs(lb-ub)-abs(Ileft[0])

    return Iup,Idown,Ileft,Iright,h,w

def binarize_image(image,threshold=70):
    image[image<=threshold] = 0
    image[image>threshold] = 255
    return image
    # mn=np.min(image)
    # mx=np.max(image)
    # image[image==mx]=255
    # if mn!=0:
    #     image[image==mn]=0
    # for i in range(1,image.shape[0]):
    #     for j in range(1,image.shape[1]):
    #         if 0<image[i,j]<255:
    #             if np.min(image[i-1:i+2,j-1:j+2])==0:
    #                 image[i,j]=0
    # return image


def resize_image(image,width,height):

    return cv2.resize(image,dsize=(width,height),interpolation=cv2.INTER_NEAREST)


if __name__=="__main__":

    # np.set_printoptions(linewidth=200,threshold=sys.maxsize)
    tesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' #required for tesseract to work
    custom_config = r'--oem 3 --psm 3' # Adding custom options
    max_height=25#max height for resizing
    max_width = 25  # max height for resizing

    inputPth=r"D:\College\Masters\Spring-2020-21\CS583\Project\images\test"
    # inputPth=r"D:\College\Masters\Spring-2020-21\CS583\HW1"

    # imgName = "a_u.png"
    # img = load_image(os.path.join(inputPth, imgName))
    # plt.imshow(img, cmap="gray")
    # plt.waitforbuttonpress(0)
    # print(img.shape)
    #
    # img=binarize_image(img,70)
    # plt.imshow(img,cmap="gray")
    # plt.waitforbuttonpress(0)

    # img=canny(img)
    # plt.imshow(img,cmap="gray")
    # plt.waitforbuttonpress(0)
    # print(water_fill5(img))

    for file in os.listdir(inputPth):
        if file.__contains__("img")==False:
            imgName = file
            img = load_image(os.path.join(inputPth,imgName))
            img=binarize_image(img,70)
            img=canny(img)
            upCap,downCap,leftCap,rightCap,h,w=water_fill6(img)
            print(imgName,upCap,downCap,leftCap,rightCap,h,w,sep=",")



    # np.savetxt(r"D:\College\Masters\Spring-2020-21\CS583\Project\images\text_comparison\numpyCanny.csv", img,
    #            delimiter=",", fmt="%d")


    # sobel_mag,sobe_theta=sobel(img)
    # np.savetxt(r"D:\College\Masters\Spring-2020-21\CS583\Project\images\text_comparison\numpySobel_grad.csv", img,delimiter=",", fmt="%d")
    # np.savetxt(r"D:\College\Masters\Spring-2020-21\CS583\Project\images\text_comparison\numpySobel_theta.csv", sobe_theta, delimiter=",", fmt="%f")