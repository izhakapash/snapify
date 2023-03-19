import cv2
import numpy as np
import os

def sharpen_image(img, amount=1, radius=1):
    # Convert the input image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to the grayscale image
    blur = cv2.GaussianBlur(gray, (2 * radius + 1, 2 * radius + 1), 0)

    # Calculate the "unsharp mask" by subtracting the blurred image from the original grayscale image
    unsharp_mask = cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)

    # Create a color image by copying the sharpened grayscale image into each color channel
    color = cv2.cvtColor(unsharp_mask, cv2.COLOR_GRAY2BGR)

    # Apply the sharpening effect to the original color image
    result = cv2.addWeighted(img, 1, color, 0.5, 0)

    return result

def sharpen_foreground(img, amount=1, radius=1):
    # Convert the input image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to the grayscale image to create a binary mask
    _, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Apply a Gaussian blur to the binary mask to create a feathered edge
    mask = cv2.GaussianBlur(mask, (2 * radius + 1, 2 * radius + 1), 0)

    # Apply the sharpening effect to the foreground only
    foreground = cv2.cvtColor(cv2.bitwise_and(img, img, mask=mask), cv2.COLOR_BGR2GRAY)
    foreground = cv2.GaussianBlur(foreground, (2 * radius + 1, 2 * radius + 1), 0)
    foreground = cv2.addWeighted(foreground, 1 + amount, cv2.GaussianBlur(foreground, (2 * radius + 1, 2 * radius + 1), 0), -amount, 0)
    foreground = cv2.cvtColor(foreground, cv2.COLOR_GRAY2BGR)
    foreground = cv2.resize(foreground, (img.shape[1], img.shape[0]))

    result = img.copy()
    result[mask == 0] = foreground[mask == 0]

    return result

def highlight_image(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to the image
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=2, sigmaY=2)

    # Subtract the blurred image from the original to create a high-pass filtered image
    highpass = gray - blur

    # Scale the high-pass filtered image to the range [0, 255]
    highpass = cv2.normalize(highpass, None, 0, 255, cv2.NORM_MINMAX)

    # Add the high-pass filtered image back to the original image to create a sharpened version
    sharpened = cv2.addWeighted(gray, 1.5, highpass, -0.5, 0)
    return sharpened

def adjust_exposure(img, average_exposure, beta):
    avg_pixel_value = np.mean(img)/128
    alpha = (avg_pixel_value/average_exposure)/2 +0.5
    img = np.uint8(np.clip((alpha * img ), 0, 255))
    return img

def adjust_contrast(img, average_contrast):
    std_pixel_value = np.std(img)/128
    level = std_pixel_value/average_contrast
    factor = (259 * (level + 255)) / (255 * (259 - level))
    table = np.array([((i / 255.0) ** factor) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def adjust_highlight(img, highlight):
    # Convert the input image to float32 data type
    img = img.astype(np.float32)

    # Calculate the maximum pixel value of the input image
    max_pixel_value = np.max(img)

    # Calculate the scale factor to adjust the highlight parameter
    scale_factor = (255.0 / max_pixel_value) * (highlight / 100.0)

    # Scale the pixel values of the image by the scale factor
    img = np.uint8(np.clip(scale_factor * img, 0, 255))

    return img

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def change_shadows(image, shadow_average):
    shadow_image = calculate_shadow_factor(image)
    shadow_factor = shadow_average/shadow_image
    # convert the image to the YUV color space, which separates the
    # intensity (luma) component from the chrominance (color) components
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)

    # apply gamma correction to the luma channel to adjust the shadows
    y = adjust_gamma(y, shadow_factor)

    # merge the adjusted luma channel back into the YUV image
    yuv = cv2.merge([y, u, v])

    # convert the YUV image back to the BGR color space
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def calculate_shadow_factor(image):
    # convert the image to the YUV color space, which separates the
    # intensity (luma) component from the chrominance (color) components
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)

    # compute the mean and standard deviation of the luma channel
    mean, std = cv2.meanStdDev(y)

    # calculate the shadow factor as the ratio of the mean to the maximum pixel value
    shadow_factor = mean[0][0] / 255.0

    return shadow_factor


def adjust_levels(image, black_level=0, white_level=255):
    # apply the black level
    if black_level > 0:
        image = np.clip(image + black_level, 0, 255).astype("uint8")

    # apply the white level
    max_level = np.max(image)
    scale_factor = white_level / max_level
    image = (image * scale_factor).astype("uint8")

    return image



def adjust_brightness(img, average_brightness):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    h = cv2.resize(h, (img.shape[1], img.shape[0]))
    s = cv2.resize(s, (img.shape[1], img.shape[0]))
    v = cv2.resize(v, (img.shape[1], img.shape[0]))

    h = h.astype(np.uint8)
    s = s.astype(np.uint8)
    v = v.astype(np.uint8)

    value = np.mean(v) - average_brightness

    v = np.where(v + value > 255, 255, v + value)
    v = np.where(v < 1, 1, v)
    v = v.astype(np.uint8)

    final_hsv = cv2.merge((h, s, v))
    result = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return result

# Define a function to measure brightness of an image
def measure_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    brightness = np.mean(v)
    return brightness

# The folder containing the images
#folder = "C:/Users/izhak/OneDrive/Desktop/yair_images"
#output = "C:/Users/izhak/OneDrive/Desktop/output_100msdcf"
folder = "C:/Users/izhak/OneDrive/Desktop/snapify/lifney"
output = "C:/Users/izhak/OneDrive/Desktop/exposure_output"

# Initialize the sum of exposures and the number of images processed
total_black_level = 0
total_white_level = 0
exposure_sum = 0
contrast_sum = 0
highlight_sum =0
shadows_sum = 0
image_count = 0
total_shadow_level = 0
total_num_pixels = 0

mean_values = []
shadow_factors = []
brightness_values = []

# Iterate over all files in the folder
for filename in os.listdir(folder):
    # Load the image
    img = cv2.imread(os.path.join(folder, filename))
    #img = sharpen_foreground(img)

    # Calculate the average pixel value of the image
    avg_pixel_value = np.mean(img)

    # Assume the exposure is proportional to the average pixel value
    exposure = avg_pixel_value / 128.0

    # Update the sum of exposures and the number of images processed
    exposure_sum += exposure
    image_count += 1

    # Calculate the standard deviation of the pixel values of the image
    std_pixel_value = np.std(img)

    # Assume the contrast is proportional to the standard deviation of the pixel values
    contrast = std_pixel_value / 128.0

    # Update the sum of contrasts and the number of images processed
    contrast_sum += contrast

    # Calculate the maximum pixel value of the image
    max_pixel_value = np.max(img)

    # Calculate the percentage of the image that is considered highlights
    highlight = (np.sum(img > max_pixel_value * 0.9) / img.size) * 100.0

    # Update the sum of highlights and the number of images processed
    highlight_sum += highlight

    shadow_factor = calculate_shadow_factor(img)

    # add the shadow factor to the list
    shadow_factors.append(shadow_factor)

    # adjust the levels to calculate the black and white levels
    image = adjust_levels(img)
    black_level = np.min(image)
    white_level = np.max(image)

    # add the black and white levels to the running totals
    total_black_level += black_level
    total_white_level += white_level

    # Measure the brightness of the image and append it to the list
    brightness = measure_brightness(img)
    brightness_values.append(brightness)






# Calculate the average exposure of all images
average_exposure = exposure_sum / image_count
# Calculate the average contrast of all images
average_contrast = contrast_sum / image_count
# Calculate the average highlight of all images
average_highlight = highlight_sum / image_count
# Calculate the average shadows of all images
average_shadow_factor = np.mean(shadow_factors)

avg_black_level = total_black_level / image_count
avg_white_level = total_white_level / image_count
# Calculate the average brightness across all images
average_brightness = np.mean(brightness_values)

# Print the result
print('The average exposure of all images is:', average_exposure)
print('The average contrast of all images is:', average_contrast)
print('The average highlight of all images is:', average_highlight)
print('The average shadows of all images is:', average_shadow_factor)
print('The average avg_black_level of all images is:', avg_black_level)
print('The average avg_white_level of all images is:', avg_white_level)
print('The average average_brightness of all images is:', average_brightness)


for filename in os.listdir(folder):
    # Load the image
    img = cv2.imread(os.path.join(folder, filename))
    #img = adjust_exposure(img, average_exposure, 0)
    img = adjust_contrast(img,average_contrast)
    #img = adjust_highlight(img,average_highlight)
    #img = change_shadows(img,average_shadow_factor)
    #img = adjust_levels(img,0,int(avg_white_level))
    #img = sharpen_foreground(img)
    #img = adjust_brightness(img,average_brightness)
    cv2.imwrite(output + '/' + filename, img)

test_path = "C:/Users/izhak/OneDrive/Desktop/test2"
test_output = "C:/Users/izhak/OneDrive/Desktop/test_output"

for filename in os.listdir(test_path):
    # Load the image
    img = cv2.imread(os.path.join(test_path, filename))
    img = adjust_exposure(img, average_exposure, 0)
    img = adjust_contrast(img,average_contrast)
    #img = adjust_highlight(img,average_highlight)
    img = change_shadows(img,average_shadow_factor)
    #img = adjust_levels(img,0,int(avg_white_level))
    #img = sharpen_foreground(img)
    img = adjust_brightness(img,average_brightness)
    cv2.imwrite(test_output + '/' + filename, img)

