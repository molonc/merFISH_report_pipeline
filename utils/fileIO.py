

import skimage.io as skio
import numpy as np
import json



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
                }
        a_file = open(coord_file, "w")
        a_file = json.dump(data, a_file)
