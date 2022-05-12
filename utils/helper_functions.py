import os
import numpy as np
import json
import re
from pathlib import Path
from typing import Callable
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import pickle
import skimage
from skimage import filters,restoration
import sys


def check_dirs(files:list)->None:
    """Checks to see if the directories for the files in the list exist. If they dont, then make those directories

    Args:
        files (list[str]): the list of files whose directory paths to create. Needs to be the full, not relative paths
        
    """
    if not type(files) == list:
        d = os.path.dirname(files)
        if not os.path.isdir(d):
            # print(files+'files' + d +'Does not exist')
            os.makedirs(d)
    else:
        for f in files:

            d = os.path.dirname(f)
            if d != "" and not os.path.isdir(d):
                # print(f+'files' + d +'Does not exist')
                os.makedirs(d)


def identity(img):
    return img


def transformWarpedImages(warped_images, transform,clip_thresh):

    pp_wi = list()
    filtered_wi = list()
    for wi in range(warped_images.shape[0]):
        _img = transform(warped_images[wi,:,:])

        pp_wi.append(_img)


    for wi in pp_wi:
        _img = np.where(wi>np.percentile(wi,clip_thresh),1,0)

        filtered_wi.append(_img)


    filtered_wi = np.stack(np.array(filtered_wi),axis=0)

    return filtered_wi

def getBrightPixelCount(filtered_warped_images):
    
    bright_pix_count = filtered_warped_images.sum(axis=0)

    return bright_pix_count

def getBarcodesFromPixelStack(codebook,pixel_stack,filtered_wi):



    codebook_bits = np.array(codebook.barcode_arrays)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree",p=1).fit(codebook_bits)
    pixel_stack = pixel_stack[np.newaxis,:,:]
    bit_map = dict()
    barcode_map = dict()
    dist_map = dict()
    bit_vector_map = dict()
    print(f'Pixel Stack Shape={pixel_stack.shape}')
    print(f'Filtered WI Shape={filtered_wi.shape}')
    
    bit_map[3]= np.multiply(pixel_stack==3,filtered_wi)
    bit_map[4]= np.multiply(pixel_stack==4,filtered_wi)
    bit_map[5]= np.multiply(pixel_stack==5,filtered_wi)
    print(f'Bitmap 5 Shape={bit_map[5].shape}')
    bit_vector_map[3] = np.reshape(bit_map[3],(16,-1)).T
    bit_vector_map[4] = np.reshape(bit_map[4],(16,-1)).T
    bit_vector_map[5] = np.reshape(bit_map[5],(16,-1)).T

    # Look at distances for 3 pixel barcodes
    dist_map[3], barcode_map[3] = nbrs.kneighbors(bit_vector_map[3])
    dist_map[4], barcode_map[4] = nbrs.kneighbors(bit_vector_map[4])
    dist_map[5], barcode_map[5] = nbrs.kneighbors(bit_vector_map[5])
    
    return bit_map,bit_vector_map,dist_map,barcode_map,codebook_bits
    
def filterDetections(barcode_ids,distances,filter_val):
    return barcode_ids[distances==filter_val]


def findErrorBits(bright_pix,codebook_bits, dist, barcode_ids,filter_val=0):

    barcode_subset = barcode_ids[dist==filter_val]
    bright_pix_subset = bright_pix[np.where(dist==filter_val)[0],:]
    codebook_subset = codebook_bits[barcode_subset]
    error_bit_locs = np.argmax(np.abs(bright_pix_subset - codebook_subset),axis=1)
    
    return error_bit_locs

def performDumbExtraction(warped_images, codebook,clip_thresh=98.5):
    transform=identity
    
    filtered_warped_images = transformWarpedImages(warped_images, transform,clip_thresh)
    
    bright_pixel_count = getBrightPixelCount(filtered_warped_images)
    
    bit_map,bit_vector_map,dist_map,barcode_map,codebook_bits = getBarcodesFromPixelStack(codebook,bright_pixel_count,filtered_warped_images)
    
    error_bit_map = dict()
    error_bit_map[3] = findErrorBits(bit_vector_map[3],codebook_bits,dist_map[3],barcode_map[3],filter_val=1)

    error_bit_map[4] = findErrorBits(bit_vector_map[4],codebook_bits,dist_map[4],barcode_map[4],filter_val=0)

    error_bit_map[5] = findErrorBits(bit_vector_map[5],codebook_bits,dist_map[5],barcode_map[5],filter_val=1)
    
    detection_number_map = dict()
    detection_number_map[3]=dist_map[3][dist_map[3]==1].sum()
    detection_number_map[4]=(1+dist_map[4][dist_map[4]==0]).sum()
    detection_number_map[5]=dist_map[5][dist_map[5]==1].sum()
    
    return filtered_warped_images, bright_pixel_count, bit_map, bit_vector_map, dist_map, barcode_map, error_bit_map, detection_number_map,codebook_bits

