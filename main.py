from pytesseract import pytesseract as tesseract
import numpy as np
from PIL import Image
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from water_fill_test import water_fill6

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

def rotate_image(image,degree=0):
    if degree==0:
        return image
    if abs(degree)==180:
        return cv2.rotate(image,cv2.ROTATE_180)
    if degree==90:
        return cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
    if degree==-90:
        return cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)

    return cv2.rotate(image,cv2.ROTATE_180)

def save_image(image,pth,name):
    cv2.imwrite(os.path.join(pth,name),image)

def tesseract_detect_orientation(image):
    orientation = tesseract.image_to_osd(image,config="--oem 3 --psm 0") #detect orientation
    return orientation

def binarize_image(image,threshold=125):
    image[image<=threshold] = 0 #black
    image[image>threshold] = 255 #white
    return image

def waterFill(image,charMap,color=False):
    h, w = image.shape
    boxes = tesseract.image_to_boxes(image,lang="eng",config=r'--oem 3 --psm 3')
    up_thres=0.032
    down_thres=0.095
    left_thres=0
    right_thres=0.012

    boxed_image = image.copy()
    if color:boxed_image=cv2.cvtColor(boxed_image,cv2.COLOR_GRAY2BGR)

    upright = inverted = 0
    for b in boxes.splitlines():
        # Letter, coordinates of bottom left corner of bounding box,
        # coordinates of top right of bounding box
        bx=b.split(" ")

        #If tesseract identifies it as symmetric letter, continue
        if bx[0] in charMap.loc[:,"CharName"].values:
            rw= charMap[charMap["CharName"]==bx[0]].first_valid_index()
            if charMap.loc[rw,"Symmetric"]==True: #Symmetric column
                continue

        bx[1:]=map(int,bx[1:]) #convert coordinates to integers

        #get the character found by tesseract
        cropped_image = image[h - bx[4]:h - bx[2] + 1, bx[1]:bx[3] + 1]
        #convert any blur to either black or white pixel values
        t=143
        cropped_image_b=binarize_image(cropped_image.copy(),threshold=t)
        cropped_image_b=np.abs(cropped_image_b-255) #invert the image

        #side-by side comarison of binarized and non-binarized image
        # fig,axArr=plt.subplots(1,2)
        # axArr[0].imshow(cropped_image,cmap="gray")
        # axArr[1].imshow(cropped_image_b,cmap="gray")
        # plt.draw()
        # plt.waitforbuttonpress(0)
        # plt.close(fig)

        #Calculate capacities for character
        cib_h,cib_w=cropped_image_b.shape
        cib_area=cib_h*cib_w
        U,D,L,R,_,_=water_fill6(cropped_image_b)
        #Capacity per pixel area
        U,D,L,R=U/cib_area,D/cib_area,L/cib_area,R/cib_area

        if U<up_thres:U=0
        if D<down_thres:D=0
        if L<left_thres:L=0
        if R<right_thres:R=0

        upright+=D+R
        inverted+=U+L

        # #Calculate error in relation to all characters
        # uErr = (charMap.loc[:, "Up_Capacity"] - U) ** 2  # sq. error
        # dErr = (charMap.loc[:, "Down_Capacity"] - D) ** 2  # sq. error
        # lErr = (charMap.loc[:, "Left_Capacity"] - L) ** 2  # sq. error
        # rErr = (charMap.loc[:, "Right_Capacity"] - R) ** 2  # sq. error
        #
        # totalErr = np.sqrt(uErr + dErr + lErr + rErr)
        # totalErr.sort_values(inplace=True)

        #output image with boxes around tesseract detected characters
        boxed_image = cv2.rectangle(boxed_image, (bx[1], h - bx[2]), (bx[3], h - bx[4]), (80, 176, 0), 0)

    if upright>=inverted:
        return "upright",boxed_image,upright,inverted
    else:
        return "inverted",boxed_image,upright,inverted


if __name__=="__main__":

    tesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' #required for tesseract to work
    custom_config = r'--oem 3 --psm 3' # Adding custom options

    projectPth=os.path.dirname(os.path.abspath(__file__))
    inputPth=os.path.join(projectPth,r"images\test")
    outputPth=os.path.join(projectPth,r"images\box_results")


    #Load in character map
    charMap=pd.read_csv(os.path.join(projectPth,"characterMap.csv"),sep=",",header=0,
                        dtype={"Symmetric":np.bool_,
                               "Inverted":np.bool_,
                               "CharName":np.str},index_col="Image")

    imgName="img1_u.png" #image name

    #load image as grayscale
    img = load_image(os.path.join(inputPth,imgName))
    img=rotate_image(img,degree=0)

    res,imgBox,uprightCap,invertedCap = waterFill(img,charMap,color=True)

    #save image with boxes around chars
    save_image(imgBox,pth=outputPth,name=imgName)

    print("Result",res)
    print("Upright Total Capacity",uprightCap)
    print("Inverted Total Capacity", invertedCap)

    if res=="inverted":
        img=rotate_image(img)


    # newdata = tesseract_detect_orientation(img_rt) #detect orientation
    # print(newdata)

    s=tesseract.image_to_string(img, lang="eng", config=custom_config)
    print(s)