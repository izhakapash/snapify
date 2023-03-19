import cv2
import numpy as np
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt


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

def adjust_e_xposure(img, alpha, beta):
    img = np.uint8(np.clip((alpha * img + beta), 0, 255))
    return img

def adjust_exposure(image, exposure_adjustment):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert image to linear color space
    #linear_image = np.power(gray_image / 255.0, 2.2)

    # Compute histogram of image
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # Determine target exposure value
    midpoint = np.sum(hist * np.arange(256)) / np.sum(hist)
    target_exposure = midpoint * exposure_adjustment

    # Adjust image brightness
    adjusted_image = gray_image * (target_exposure / midpoint)

    # Adjust image contrast and tonal range
    adjusted_image = cv2.normalize(adjusted_image, None, 0, 1, cv2.NORM_MINMAX)
    adjusted_image = np.power(adjusted_image, 1.0 / 2.2)
    adjusted_image = cv2.cvtColor(adjusted_image, cv2.COLOR_GRAY2BGR)

    return (adjusted_image * 255).astype(np.uint8)

def adjust_exposure_2(img, exposure):
    # Convert the input image to float32 for more accurate calculations
    img_c = img
    img = np.float32(img)

    # Calculate the histogram of the input image
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # Calculate the cumulative distribution of the histogram
    cdf = hist.cumsum()

    # Normalize the cumulative distribution to the range [0, 1]
    cdf_normalized = cdf / cdf.max()

    # Debugging code - check for NaN values
    print(f"cdf_normalized: {cdf_normalized}")
    #print(f"cdf_normalized[min_val]: {cdf_normalized[min_val]}")
    #print(f"cdf_range: {cdf_range}")

    # Adjust the exposure by shifting the histogram to the left or right
    min_val = np.min(np.where(cdf_normalized > 0))
    max_val = np.max(np.where(cdf_normalized < 1))
    cdf_range = cdf_normalized[max_val] - cdf_normalized[min_val]
    if cdf_range <= 0:
        cdf_range = 1e-6
    hist_adj = ((cdf_normalized - cdf_normalized[min_val]) / cdf_range) ** exposure
    hist_adj = np.uint8(255 * hist_adj)
    hist_adj = cv2.resize(hist_adj, (256, 1), interpolation=cv2.INTER_LINEAR)

    print(f"hist_adj: {hist_adj}")
    print(f"np.isnan(hist_adj).any(): {np.isnan(hist_adj).any()}")

    # Apply the histogram adjustment to the image
    img_adj = cv2.LUT(img_c, hist_adj)

    # Convert the adjusted image back to uint8 format for display
    img_adj = np.uint8(np.clip(img_adj, 0, 255))
    img_adj = np.uint8(img_adj)

    return img_adj

def gamma_trans(img, gamma):
    gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img,gamma_table)

def adjust_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    table = np.array([((i / 255.0) ** factor) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def adjust_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    h = cv2.resize(h, (img.shape[1], img.shape[0]))
    s = cv2.resize(s, (img.shape[1], img.shape[0]))
    v = cv2.resize(v, (img.shape[1], img.shape[0]))

    h = h.astype(np.uint8)
    s = s.astype(np.uint8)
    v = v.astype(np.uint8)

    v = np.where(v + value > 255, 255, v + value)
    v = np.where(v < 1, 1, v)
    v = v.astype(np.uint8)

    final_hsv = cv2.merge((h, s, v))
    result = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return result

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def change_shadows(image, shadow_factor):
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


def adjust_levels(image, black_level=0, white_level=255):
    # apply the black level
    if black_level > 0:
        image = np.clip(image + black_level, 0, 255).astype("uint8")

    # apply the white level
    max_level = np.max(image)
    scale_factor = white_level / max_level
    image = (image * scale_factor).astype("uint8")

    return image


def change_levels(image, black_factor=1, white_factor=1):
    # adjust the black and white levels for each channel
    blue, green, red = cv2.split(image)
    blue = adjust_levels(blue, black_level=np.min(blue) * black_factor,
                         white_level=np.max(blue) * white_factor)
    green = adjust_levels(green, black_level=np.min(green) * black_factor,
                          white_level=np.max(green) * white_factor)
    red = adjust_levels(red, black_level=np.min(red) * black_factor,
                        white_level=np.max(red) * white_factor)

    # merge the adjusted channels back into the original image
    return cv2.merge([blue, green, red])

def correction(
        img,
        shadow_amount_percent, shadow_tone_percent, shadow_radius,
        highlight_amount_percent, highlight_tone_percent, highlight_radius,
        color_percent
):
    """
    Image Shadow / Highlight Correction. The same function as it in Photoshop / GIMP
    :param img: input RGB image numpy array of shape (height, width, 3)
    :param shadow_amount_percent [0.0 ~ 1.0]: Controls (separately for the highlight and shadow values in the image) how much of a correction to make.
    :param shadow_tone_percent [0.0 ~ 1.0]: Controls the range of tones in the shadows or highlights that are modified.
    :param shadow_radius [>0]: Controls the size of the local neighborhood around each pixel
    :param highlight_amount_percent [0.0 ~ 1.0]: Controls (separately for the highlight and shadow values in the image) how much of a correction to make.
    :param highlight_tone_percent [0.0 ~ 1.0]: Controls the range of tones in the shadows or highlights that are modified.
    :param highlight_radius [>0]: Controls the size of the local neighborhood around each pixel
    :param color_percent [-1.0 ~ 1.0]:
    :return:
    """
    shadow_tone = shadow_tone_percent * 255
    highlight_tone = 255 - highlight_tone_percent * 255

    shadow_gain = 1 + shadow_amount_percent * 6
    highlight_gain = 1 + highlight_amount_percent * 6

    # extract RGB channel
    height, width = img.shape[:2]
    img = img.astype(np.float64)
    img_R, img_G, img_B = img[..., 2].reshape(-1), img[..., 1].reshape(-1), img[..., 0].reshape(-1)

    # The entire correction process is carried out in YUV space,
    # adjust highlights/shadows in Y space, and adjust colors in UV space
    # convert to Y channel (grey intensity) and UV channel (color)
    img_Y = .3 * img_R + .59 * img_G + .11 * img_B
    img_U = -img_R * .168736 - img_G * .331264 + img_B * .5
    img_V = img_R * .5 - img_G * .418688 - img_B * .081312

    # extract shadow / highlight
    shadow_map = 255 - img_Y * 255 / shadow_tone
    shadow_map[np.where(img_Y >= shadow_tone)] = 0
    highlight_map = 255 - (255 - img_Y) * 255 / (255 - highlight_tone)
    highlight_map[np.where(img_Y <= highlight_tone)] = 0

    # // Gaussian blur on tone map, for smoother transition
    if shadow_amount_percent * shadow_radius > 0:
        # shadow_map = cv2.GaussianBlur(shadow_map.reshape(height, width), ksize=(shadow_radius, shadow_radius), sigmaX=0).reshape(-1)
        shadow_map = cv2.blur(shadow_map.reshape(height, width), ksize=(shadow_radius, shadow_radius)).reshape(-1)

    if highlight_amount_percent * highlight_radius > 0:
        # highlight_map = cv2.GaussianBlur(highlight_map.reshape(height, width), ksize=(highlight_radius, highlight_radius), sigmaX=0).reshape(-1)
        highlight_map = cv2.blur(highlight_map.reshape(height, width), ksize=(highlight_radius, highlight_radius)).reshape(-1)

    # Tone LUT
    t = np.arange(256)
    LUT_shadow = (1 - np.power(1 - t * (1 / 255), shadow_gain)) * 255
    LUT_shadow = np.maximum(0, np.minimum(255, np.int_(LUT_shadow + .5)))
    LUT_highlight = np.power(t * (1 / 255), highlight_gain) * 255
    LUT_highlight = np.maximum(0, np.minimum(255, np.int_(LUT_highlight + .5)))

    # adjust tone
    shadow_map = shadow_map * (1 / 255)
    highlight_map = highlight_map * (1 / 255)

    iH = (1 - shadow_map) * img_Y + shadow_map * LUT_shadow[np.int_(img_Y)]
    iH = (1 - highlight_map) * iH + highlight_map * LUT_highlight[np.int_(iH)]
    img_Y = iH

    # adjust color
    if color_percent != 0:
        # color LUT
        if color_percent > 0:
            LUT = (1 - np.sqrt(np.arange(32768)) * (1 / 128)) * color_percent + 1
        else:
            LUT = np.sqrt(np.arange(32768)) * (1 / 128) * color_percent + 1

        # adjust color saturation adaptively according to highlights/shadows
        color_gain = LUT[np.int_(img_U ** 2 + img_V ** 2 + .5)]
        w = 1 - np.minimum(2 - (shadow_map + highlight_map), 1)
        img_U = w * img_U + (1 - w) * img_U * color_gain
        img_V = w * img_V + (1 - w) * img_V * color_gain

    # re convert to RGB channel
    output_R = np.int_(img_Y + 1.402 * img_V + .5)
    output_G = np.int_(img_Y - .34414 * img_U - .71414 * img_V + .5)
    output_B = np.int_(img_Y + 1.772 * img_U + .5)

    output = np.row_stack([output_B, output_G, output_R]).T.reshape(height, width, 3)
    output = np.minimum(output, 255).astype(np.uint8)
    return output


# Load an image
input_image = "C:/Users/izhak/OneDrive/Desktop/snapify/lifney/IMG_0275.jpg"
output = "C:/Users/izhak/OneDrive/Desktop/lifney/output"
img = cv2.imread(input_image)

# Adjust the exposure using alpha and beta values
alpha = 0.3
beta = -120
#img = adjust_exposure_2(img, 1.5)

img = gamma_trans(img, 0.05)

#img = adjust_contrast(img, 70)

#img = adjust_brightness(img, -20)

#img = change_shadows(img, 3)

#img = correction(img=img,shadow_amount_percent=0.1, shadow_tone_percent=0.1, shadow_radius=0,
      #  highlight_amount_percent=0.7, highlight_tone_percent=0.7, highlight_radius=0,
       # color_percent=0)

black_factor = 0.99
white_factor = 1



#img = change_levels(img, black_factor, white_factor)
#img = np.abs(img)
#img =img.astype(np.uint8)

#image = img_as_float(data.moon())
#img = exposure.adjust_gamma(img, 0.1)
#img = exposure.adjust_log(img,6)
#img = exposure.adjust_sigmoid(img,0.5,inv=False)
# Output is darker for gamma > 1
#img.mean() > gamma_corrected.mean()
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
equ = cv2.equalizeHist(gray_image)

#show_histogram(img,output)
plt.plot(equ, color='b')
plt.xlim([0, 256])
plt.savefig(output +'/equ_histogram.png')
# Save the adjusted image
cv2.imwrite(output +'/test_image.jpg', img)
