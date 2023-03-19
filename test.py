import cv2
import numpy as np
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import os
import tqdm
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as TF



def show_histogram(img,output,filename):
    b, g, r = cv2.split(img)
    r_hist = cv2.calcHist([r], [0], None, [256], [0, 256])
    g_hist = cv2.calcHist([g], [0], None, [256], [0, 256])
    b_hist = cv2.calcHist([b], [0], None, [256], [0, 256])
    plt.plot(r_hist, color='r')
    plt.plot(g_hist, color='g')
    plt.plot(b_hist, color='b')
    plt.xlim([0, 256])
    plt.savefig(output +'/'+ filename+'histogram.png')
    #plt.show()

def adjust_contrast(img, average_contrast):
    std_pixel_value = np.std(img)/128
    level = std_pixel_value/average_contrast
    factor = (259 * (level + 255)) / (255 * (259 - level))
    table = np.array([((i / 255.0) ** factor) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def changh_contrast(img,avg_var_all):
    lap_var = np.var(cv2.Laplacian(gray_img, cv2.CV_64F))

    # Calculate the scaling factor to adjust the contrast
    scale_factor = np.sqrt(avg_var_all / lap_var)

    # Apply the contrast adjustment to the image
    adj_img = cv2.convertScaleAbs(img, alpha=scale_factor, beta=0)
    return adj_img



def gamma_trans(img, gamma):
    gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img,gamma_table)

def cal_contrast(gray_img):
    # Apply Laplacian filtering
    laplacian_img = cv2.Laplacian(gray_img, cv2.CV_64F)

    # Calculate the variance of the Laplacian
    lap_var = np.var(laplacian_img)
    return lap_var


def change_exposure(img,target_exposure):
    if abs(exposure_score(img)-target_exposure) <4:
        return img
    if (exposure_score(img) == 255) or (exposure_score(img)==0):
        return img
    while exposure_score(img) < target_exposure:
        img = gamma_trans(img,0.9)
    while exposure_score(img) > target_exposure:
        print(exposure_score(img))
        img = gamma_trans(img,1.1)
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
    #plt.savefig("C:/Users/izhak/OneDrive/Desktop/exposure_output/"+filename+ ".png")

    # Display the results
    #plt.show()
    #print("Exposure value: ", exposure)
    return exposure

def adjust_temperature(img, temperature):
    # Define the lookup table
    if temperature > 0:
        lookup_table = np.array([((i/255)**(1/temperature))*255 for i in np.arange(0, 256)]).astype('uint8')
    else:
        lookup_table = np.array([((i/255)**(temperature))*255 for i in np.arange(0, 256)]).astype('uint8')

    # Apply the lookup table to the image
    result = cv2.LUT(img, lookup_table)

    return result





# The folder containing the images
#folder = "C:/Users/izhak/OneDrive/Desktop/yair_images"
#output = "C:/Users/izhak/OneDrive/Desktop/output_100msdcf"
#folder = "C:/Users/izhak/OneDrive/Desktop/snapify/lifney"
folder = "C:/Users/izhak/OneDrive/Desktop/snapify/lifney"
output = "C:/Users/izhak/OneDrive/Desktop/snapify/exposure_output"

exposure_list = []
avg_pixel_list = []
var_list = []
contrast_values = []
for idx,filename in enumerate(os.listdir(folder)):
    # Load the image
    img = cv2.imread(os.path.join(folder, filename))
    # convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Calculate the average pixel value of the image
    exposure_list.append(exposure_score(img))
    avg_pixel_list.append(round(np.mean(img)))
    std_pixel_value = np.std(img)

    var_list.append(cal_contrast(gray_img))



average_exposure = np.average(exposure_list)
average_pixels = np.average(avg_pixel_list)
average_contrast = np.mean(var_list)
avg_contrast = np.mean(contrast_values)




for filename in tqdm.tqdm(os.listdir(folder)):
    # Load the image
    img = cv2.imread(os.path.join(folder, filename))
    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = change_exposure(img,average_exposure)
    #img = adjust_contrast(img, average_contrast)
    #img = changh_contrast(img,average_contrast)
    #equalized_img = cv2.equalizeHist(img)
    # Adjust the temperature of the image

    temperature_factor = 0.9
    b, g, r = cv2.split(img)
    max_value = np.max(img)
    b = np.clip(b * temperature_factor, 0, max_value).astype(np.uint8)
    r = np.clip(r / temperature_factor, 0, max_value).astype(np.uint8)
    temp_adjusted_img = cv2.merge((b, g, r))

    img = exposure.adjust_sigmoid(img, 0.3, inv=False)
    img_tensor = TF.to_tensor(Image.open(folder + '/' + filename))
    adjusted_tensor = TF.adjust_contrast(img_tensor, 100)
    adjusted_tensor = img_tensor.numpy().transpose((1, 2, 0))
    #show_histogram(img,output,filename)
    #cv2.imshow("",temp_adjusted_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite(output + '/' + filename,temp_adjusted_img)
