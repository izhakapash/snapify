import cv2
import numpy as np
import os
from matplotlib import pyplot as plt



def text (image,x,y,x2,y2,theta):
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (x,y)

    # fontScale
    fontScale = 1

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 1

    # Using cv2.putText() method
    image = cv2.putText(image, f"{round(theta)}", org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    org = (x2,y2)
    #image = cv2.putText(image, f"{ x2, y2}", org, font,
    #                    fontScale, color, thickness, cv2.LINE_AA)
    return image


def get_roll_angle_sobel(image,output,filename):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold, binary_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    sobel = cv2.Sobel(binary_img, cv2.CV_64F, 0, 1, ksize=3)
    s_64 = np.absolute(sobel)
    s = np.uint8(s_64)
    cv2.imwrite(output + '/' + filename + 'binary.png', binary_img)
    cv2.imwrite(output + '/' + filename + 'sobely.png', s)
    lines = cv2.HoughLinesP(s, 1, np.pi/180,threshold=30,minLineLength=100,maxLineGap=20)
    roll_list = []
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
            theta = np.arctan((y2-y1)/(x2-x1))
            theta = np.rad2deg(theta)
            if abs(theta) < 7:
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                image = text(image,x1,y1,x2,y2,theta)
                roll_list.append(theta)
    cv2.imwrite(output + '/' + filename + 'sobel_line_image.png', image)
    #cv2.imshow("",image)
    #cv2.waitKey(0)
    #if roll_list is None:
     #   return
    roll_angle = np.mean(roll_list)
    roll_var = np.var(roll_list)
    return roll_angle, roll_var


def get_roll_angle_canny(image,output,filename):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(gray, 50, 250, apertureSize=3)
    laplacian = cv2.Laplacian(blur,cv2.CV_64F,ksize=5)
    s_64 = np.absolute(edges)
    s = np.uint8(s_64)
    cv2.imwrite(output+'/' + filename + 'original.png', gray)
    cv2.imwrite(output+'/' + filename + 'edge.png', edges)
    cv2.imwrite(output + '/' + filename + 'laplacian.png', laplacian)
    lines = cv2.HoughLinesP(s, 1, np.pi/180,threshold=30,minLineLength=100,maxLineGap=20)
    roll_list = []
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
            theta = np.arctan((y2-y1)/(x2-x1))
            theta = np.rad2deg(theta)
            if abs(theta) < 7:
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                image = text(image,x1,y1,x2,y2,theta)
                roll_list.append(theta)
    cv2.imwrite(output + '/' + filename + 'line_image.png', image)
    #cv2.imshow("",image)
    #cv2.waitKey(0)
    if roll_list is None:
        return
    roll_angle = np.mean(roll_list)
    roll_var = np.var(roll_list)
    return roll_angle, roll_var


def choose_roll(roll_sobel,roll_canny):
    pass


def rotation(img,angle):

    if np.isnan(angle):
        return img
    theta = angle
    # Get the image dimensions
    (h, w) = img.shape[:2]
    # Calculate the center of the image
    center = (w // 2, h // 2)
    # Define the rotation matrix
    M = cv2.getRotationMatrix2D(center, theta, 1.0)
    # Perform the rotation
    rotated = cv2.warpAffine(img, M, (w, h))
    x_offset = int(w * np.sin(abs(np.deg2rad(theta))))
    y_offset = int(h * np.sin(abs(np.deg2rad(theta))))
    print(y_offset,x_offset)
    cropped = rotated[y_offset:h-y_offset,x_offset:w-x_offset]
    # Resize the cropped image to remove black space
    resized_rotated = cv2.resize(cropped, (w, h))
    cv2.imwrite("C:/Users/izhak/OneDrive/Desktop/snapify/output_roll_canny/" + filename + 'resized_rotated.png', resized_rotated)
    #cv2.imshow('Resized Image', resized_rotated)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return resized_rotated


folder = "C:/Users/izhak/OneDrive/Desktop/snapify/lifney"
input_image = "C:/Users/izhak/OneDrive/Desktop/snapify/lifney/IMG_0288.jpg"
output = "C:/Users/izhak/OneDrive/Desktop/snapify/output_roll"

for idx,filename in enumerate(os.listdir(folder)):
    # Load the image
    image = cv2.imread(os.path.join(folder, filename))
    roll_canny = get_roll_angle_canny(image,output,filename)
    print(roll_canny,"canny",filename)
    roll_sobel = get_roll_angle_sobel(image,output,filename)
    print(roll_sobel,"sobel",filename)
    #roll = choose_roll(roll_sobel,roll_canny)
    rotation(image,roll_canny[0])