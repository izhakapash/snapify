#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy
import os
import glob


def run_testing_on_dataset(dataset_dir,output,threshold):
    blur_list = []
    img_list = os.listdir(dataset_dir)
    for ind, image_name in enumerate(img_list):
        print("Blurry Image Prediction: %d / %d images processed.." % (ind, len(img_list)))

        # Read the image
        img_path = dataset_dir + "/" + image_name
        img = cv2.imread(str(img_path), 0) # grayscale image
        image = cv2.imread(str(img_path)) # color image
        _,score,prediction = estimate_blur(img, threshold=threshold)
        #cv2.imwrite(output + "/noam_blur_output/" + str(score) + image_name, image)
        if prediction:
            cv2.imwrite(output+ "/20/" + image_name,image)
        else:
            blur = image_name,score
            blur_list.append(blur)
    print(blur_list)
    return blur_list


def fix_image_size(image: numpy.array, expected_pixels: float = 2E6):
    ratio = numpy.sqrt(expected_pixels / (image.shape[0] * image.shape[1]))
    return cv2.resize(image, (0, 0), fx=ratio, fy=ratio)


def estimate_blur_norm(image: numpy.array, threshold: float):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    score = numpy.var(blur_map)
    score_norm = (score - 0) / (20 - 0)  # linearly normalize score between 0 and 1
    return blur_map, score, (score_norm < threshold)


def estimate_blur(image: numpy.array, threshold: int = 37):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    score = numpy.var(blur_map)
    return blur_map, score, bool(score > threshold)


def pretty_blur_map(blur_map: numpy.array, sigma: int = 5, min_abs: float = 0.5):
    abs_image = numpy.abs(blur_map).astype(numpy.float32)
    abs_image[abs_image < min_abs] = min_abs

    abs_image = numpy.log(abs_image)
    cv2.blur(abs_image, (sigma, sigma))
    return cv2.medianBlur(abs_image, sigma)


folder = "C:/Users/izhak/OneDrive/Desktop/snapify/lifney"
output = "C:/Users/izhak/OneDrive/Desktop/snapify"

run_testing_on_dataset(folder,output=output,threshold=20)
