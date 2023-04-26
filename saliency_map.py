"""
Created on Wed April 26 2023
@author: yamadaaiki
"""

import argparse
import os
import cv2
from matplotlib import pyplot as plt


def make_saliency_map(input_image_data):
    
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    _, saliency_map = saliency.computeSaliency(input_image_data)
    i_saliency = (saliency_map * 255).astype("uint8")
    
    return i_saliency


def main():
    
    parser = argparse.ArgumentParser(description="Task for saliency map")
    parser.add_argument('input_image_path', help="path for image data")
    parser.add_argument('output_path', help='output folder path')
    parser.add_argument('save_saliency_map', help='if you want to save saliceny map.',
                        action='store_true')
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.input_image_path):
        raise FileNotFoundError("[Error] the image file:{args.input_iamge_path} in not found.")
    
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)
    
    image_data = cv2.imread(args.input_image_path, 1)
    
    # cv2.imwrite(f'{args.output_path}/sample2.jpg', image_data)
    
    saliency_map = make_saliency_map(image_data)
    
    # image save
    if args.save_saliency_map:
        cv2.imwrite(args.output_path, saliency_map)
    
    # show the original, saliency map, and mixed images.


if __name__ == '__main__':
    main()