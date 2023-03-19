import cv2
import numpy as np
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import os


def show_histogram(img,output):
    b, g, r = cv2.split(img)
    r_hist = cv2.calcHist([r], [0], None, [256], [0, 256])
    g_hist = cv2.calcHist([g], [0], None, [256], [0, 256])
    b_hist = cv2.calcHist([b], [0], None, [256], [0, 256])
    plt.plot(r_hist, color='r')
    plt.plot(g_hist, color='g')
    plt.plot(b_hist, color='b')
    plt.xlim([0, 256])
    plt.savefig(output +'/histogram.png')
    plt.show()


def adjust_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    table = np.array([((i / 255.0) ** factor) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def gamma_trans(img, gamma):
    gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img,gamma_table)


def change_exposure(img,target_exposure):
    if abs(exposure_score(img)-target_exposure) <1.1:
        return img
    if (exposure_score(img) == 255) or (exposure_score(img)==0):
        print ("exposure")
        return img
    while exposure_score(img) < target_exposure:
        img = gamma_trans(img,0.97)
    while exposure_score(img) > target_exposure:
        img = gamma_trans(img,1.03)
    return img


def exposure_score(img):
    hist, bins = np.histogram(img.ravel(), 256, [0, 256])

    # Calculate the cumulative distribution function (CDF)
    cdf = hist.cumsum()
    #cdf_normalized = cdf * hist.max() / cdf.max()
    cdf_normalized = cdf / cdf.max()

    # Find the exposure value
    exposure = np.argmax(cdf_normalized >= 0.5)

    # Plot the histogram and CDF
    #plt.subplot(121)
    #plt.hist(img.ravel(), 256, [0, 256])
    #plt.title('Histogram')
    #plt.subplot(122)
    #plt.plot(cdf_normalized, color='b')
    #plt.axvline(x=exposure, color='r', linestyle='--')
    #plt.title('CDF')

    # Display the results
    #plt.show()
    print("Exposure value: ", exposure)
    return exposure



# The folder containing the images
#folder = "C:/Users/izhak/OneDrive/Desktop/yair_images"
#output = "C:/Users/izhak/OneDrive/Desktop/output_100msdcf"
folder = "C:/Users/izhak/OneDrive/Desktop/snapify/lifney"
output = "C:/Users/izhak/OneDrive/Desktop/exposure_output"

exposure_list = []
for idx,filename in enumerate(os.listdir(folder)):
    # Load the image
    img = cv2.imread(os.path.join(folder, filename))
    # Calculate the average pixel value of the image
    exposure_list.append(exposure_score(img))

average_exposure = np.average(exposure_list)
for filename in os.listdir(folder):
    # Load the image
    img = cv2.imread(os.path.join(folder, filename))
    img = change_exposure(img,average_exposure)
    #img = adjust_contrast(img,average_contrast)
    #img = adjust_highlight(img,average_highlight)
    #img = change_shadows(img,average_shadow_factor)
    #img = adjust_levels(img,0,int(avg_white_level))
    #img = sharpen_foreground(img)
    #img = adjust_brightness(img,average_brightness)
    cv2.imwrite(output + '/' + filename, img)


