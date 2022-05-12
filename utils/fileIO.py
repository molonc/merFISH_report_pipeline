
import csv
import re
import skimage.io as skio
import numpy as np
import json
import pandas as pds

import matplotlib.colors as mcolors


base_colors = list(mcolors.BASE_COLORS)

def create_image_stack(image_format,fov,z,irs,wvs,out_file,coord_file):

        # if any parameter is a single string, then make it into an iterable
        # list object

        if not isinstance(irs,list):
            irs=[irs]
        if not isinstance(wvs,list):
            wvs=[wvs]
        
        # Extract a test image to get size parameter out of
        test_img = skio.imread(
            image_format.format(wv=wvs[0],fov=fov,ir=irs[0],z=z)
        )
        
        xyshape = test_img.shape
        x =  xyshape[1]
        y = xyshape[0]

       
        # Load in all the data

         # Pre-allocate the image stack
        img_stack = np.zeros((y,x,
                            len(wvs),
                            len(irs),
                             )
                            )
        for iir,ir in enumerate(irs):
            for iwv,wv in enumerate(wvs):

                img_stack[:,:,iwv,iir] = skio.imread(
                    image_format.format(wv=wv,fov=fov,ir=ir,z=z)
                )
        #Drop it all into an npy file
        np.save(out_file,img_stack)

        #Save all the coordinates for the dimensions into a seperate file
        data = {"y":y,
                "x":x,
                "wvs":wvs,
                "irs":irs,
                "fovs":[fov],
                "zs":[z]
                }
        a_file = open(coord_file, "w")
        a_file = json.dump(data, a_file)

class Codebook:
    def __init__(self, codebook_path):
        # TODO this could probably work well as a pandas df
        self.names = []
        self.ids = []
        self.barcode_strings = []
        with open(codebook_path, encoding="utf8") as csv_file:
            codebook_reader = csv.reader(csv_file)
            for i, row in enumerate(codebook_reader):
                if i == 0:
                    self.version = row[1]
                elif i == 1:
                    self.codebook_name = row[1]
                elif i == 2:
                    self.bit_names = row[1:]
                elif i >= 4:
                    self.names.append(row[0])
                    self.ids.append(row[1])
                    self.barcode_strings.append(row[2])
        self.barcode_arrays = [np.array([int(char) for char in re.sub(r"\s", "", barcode)], dtype="uint8") for barcode in
                               self.barcode_strings]
        # pickle.dump(self, open(pickle_path, 'wb', pickle.HIGHEST_PROTOCOL))

    def __len__(self):
        return len(self.names)

    def normalize_barcode(self, barcode_array):
        if np.sum(barcode_array) == 0:
            return barcode_array
        return barcode_array / np.sqrt(np.sum(barcode_array ** 2))

    def get_weighted_barcodes(self):
        magnitudes = [np.sqrt(sum(barcodes ** 2)) for barcodes in self.barcode_arrays]
        return [(self.barcode_arrays[i] / magnitudes[i]).astype("float16") for i in range(len(self.barcode_arrays))]

    def get_single_bit_error_matrix(self, barcode_id):
        barcode_array = self.barcode_arrays[barcode_id]
        bit_error_matrix = [self.normalize_barcode(barcode_array)]
        for i in range(len(barcode_array)):
            corrected_barcode = barcode_array.copy()
            corrected_barcode[i] = np.logical_not(corrected_barcode[i])
            bit_error_matrix.append(self.normalize_barcode(corrected_barcode))
        return np.array(bit_error_matrix)

def read_table(file):
    """
    Reads a file into a data frame based on it's extension
    :param file: the file to read
    :return: the pandas DataFrame
    """
    ext = file.split(".")[-1]
    if ext == "csv":
        df = pds.read_csv(file)
    elif ext == "tsv":
        df = pds.read_csv(file, '\t')
    elif ext in {"xls", "xlsx", "xlsm", "xlsb"}:
        df = pds.read_excel(file)
    else:
        raise ValueError("Unexpected file extension")
    return df

