

import skimage.io as skio
import numpy as np
import json



import matplotlib.colors as mcolors


base_colors = list(mcolors.BASE_COLORS)

def create_image_stack(image_format,wvs,fovs,irs,zs,out_file,coord_file):

        # if any parameter is a single string, then make it into an iterable
        # list object
        if not isinstance(fovs,list):
            fovs=[fovs]
        if not isinstance(irs,list):
            irs=[irs]
        if not isinstance(wvs,list):
            wvs=[wvs]
        if not isinstance(zs,list):
            zs=[zs]
        
        # Extract a test image to get size parameter out of
        test_img = skio.imread(
            image_format.format(wv=wvs[0],fov=fovs[0],ir=irs[0],z=zs[0])
        )
        
        xyshape = test_img.shape
        x =  xyshape[1]
        y = xyshape[0]

        # Pre-allocate the image stack
        img_stack = np.zeros((y,x,
                                    len(wvs),
                                    len(irs),
                                    len(fovs),
                                    len(zs)))
        
        # Load in all the data
        for iz,z in enumerate(zs):
            for ifov,fov in enumerate(fovs):
                for iir,ir in enumerate(irs):
                    for iwv,wv in enumerate(wvs):
                        
                        img_stack[:,:,iwv,iir,ifov,iz] = skio.imread(
                            image_format.format(wv=wv,fov=fov,ir=ir,z=z)
                        )
        #Drop it all into an npy file
        np.save(out_file,img_stack)
        
        #Save all the coordinates for the dimensions into a seperate file
        data = {"y":y,
                "x":x,
                "wvs":wvs,
                "irs":irs,
                "fovs":fovs,
                "zs":zs
                }
        a_file = open(coord_file, "w")
        a_file = json.dump(data, a_file)
    