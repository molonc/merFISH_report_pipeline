import numpy as np
from functools import partial
from skimage import restoration,filters
import skimage.io as skio
from skimage.exposure import equalize_adapthist
import skimage
from skimage import registration
import scipy.ndimage as ndi
import pandas as pd
def read_table(file):
    """
    Reads a file into a data frame based on it's extension
    :param file: the file to read
    :return: the pandas DataFrame
    """
    ext = file.split(".")[-1]
    if ext == "csv":
        df = pd.read_csv(file)
    elif ext == "tsv":
        df = pd.read_csv(file, '\t')
    elif ext in {"xls", "xlsx", "xlsm", "xlsb"}:
        df = pd.read_excel(file)
    else:
        raise ValueError("Unexpected file extension")
    return df

def load_data_organization(data_org_file):
    """Loads the data organization file as a list of dictionaries
     based on the extension

    Args:
        data_org_file: a path to a data organaization in the standard format
    
    Returns:
        data_org: a list of ditionaries containing each imaging rounds information
    """
    return read_table(data_org_file).to_dict("records")

def warp_image(data_org, image_stack,bit_num):
    
    warped_image_stack = []
    for bit_num in range(bit_num):
        data_org_row = data_org[bit_num]
        wv_idx =int(data_org_row["frame"]) - 1
        ir_idx = int(data_org_row["imagingRound"])
        im = image_stack[:,:,wv_idx,ir_idx]
        
        warped_image_stack.append(im)

    warped_imgs = np.stack(warped_image_stack,axis=0)

    return warped_imgs    
    

def register_stack(image_stack):
    # image stack is a 4d numpy array
    # (x,y,wv,ir)
    # take out the 2nd wavelength [should be 561] and then register all the images with respect to the first imaging round
    fiducial_wv = 1 # wavelength index for beads channel
    ref_img = image_stack[:,:,fiducial_wv,0]
    registered_images = image_stack.copy()
    for ir_idx in range(1,image_stack.shape[3]):
        img = image_stack[:,:,fiducial_wv,ir_idx]
        shift, _, _ = registration.phase_cross_correlation(
            ref_img, img, upsample_factor=100
        )
        for wv_idx in range(image_stack.shape[2]):
            _img = image_stack[:,:,wv_idx,ir_idx]
            registered_images[:,:,wv_idx,ir_idx] = ndi.shift(_img,shift)
    return registered_images

# Deconvolute Images
def _deconvolute(image_stack,out_file):
    """ Deconvolutes image to sharpen features with filtering and richardson lucy restoration
    Args:
        raw_image: 2D image
        dtype: the dtype of the raw_image being passed in
    
    Returns:
        img: filtered and restored image. This image maintains the original dtype
    """

    _imgstack = np.load(image_stack) #(y,x,wv,ir)
    d_imgstack = _imgstack.copy()
    for iir in range(_imgstack.shape[3]):
        for iwv in range(_imgstack.shape[2]):
            if iwv==1:
                continue
            raw_image = _imgstack[:,:,iwv,iir]

            img = raw_image.astype("uint16")
            # High pass filtering
            filt = (
                filters.gaussian(
                    img, sigma=3, mode="nearest", truncate=2, preserve_range=True
                )
                .round()
                .astype("uint16")
            )
            img = np.subtract(img, filt, where=img > filt, out=np.zeros_like(img))

            # Point spread function deconvolution
            psf = _gaussian_kernel((10, 10), 2)

            np.where(img == np.nan, 0, img)
            img = restoration.richardson_lucy(img, psf, iterations=20, clip=False)

            # Low pass filtering
            img = filters.gaussian(
                img,
                sigma=1,
                output=None,
                cval=0,
                multichannel=False,
                preserve_range=True,
                truncate=4.0,
            )

            d_imgstack[:,:,iwv,iir] = img
    
    np.save(out_file,d_imgstack)


def _get_psf(sigma):
    """ Gets a point spread function for a gaussian kernel
    Args:
        sigma: variance
    
    Returns:
        gaussian_kernel
    """
    kernel_size = int(2 * np.ceil(2 * sigma) + 1)
    return _gaussian_kernel(shape=(kernel_size, kernel_size), sigma=sigma)


def _gaussian_kernel(shape=(3, 3), sigma=0.5):
    """ Generates a gaussian kernel
    Args:
        shape: kernel dimensions
        sigma: variance
    
    Returns:
        kernel: a gaussian kernel
    """
    m, n = [int((ss - 1.0) / 2.0) for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    kernel = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    kernel[kernel < np.finfo(kernel.dtype).eps * kernel.max()] = 0
    sumh = kernel.sum()
    if sumh != 0:
        kernel /= sumh
    return kernel




def deconvolve_img(img,psf=None):

    if psf is None:
        psf = getPSF()
    
    #perform high pass filtering first
    
    
    
    
    deconvolved_RL = restoration.richardson_lucy(img, psf, num_iter=10)

    return deconvolved_RL

def equalize_hist(img):
    

    clip_limit=0.05
    kernel_size = 10

    _img = equalize_adapthist(img,kernel_size=kernel_size,clip_limit=clip_limit)

    return _img

def getPSF(sig=None,size=5):
    if sig is None:
        sig = size/6

    x,y = np.mgrid[-int(size//2):int(size//2),
            -int(size//2):int(size//2)]


    return gaussianPSF(x,y,sig)


def gaussianPSF(x,y,sig):
    t = np.array([[x],[y]])

    if not isinstance(sig,np.array):
        sigma = np.array([[sig]])
    else:
        sigma = sig

    if sigma.shape[0] != sigma.shape[1] or len(sigma.shape)!=2:
        print('sigma is baddddd')
    
    t = -0.5*t.T@np.linalg.inv(sigma)@t

    return np.exp(t)



def maskImages(d_img_file,out_mask,percentile=98.5):
    d_imgstack = np.load(d_img_file)
    mask_imgstack = np.zeros_like(d_imgstack)
    for iir in range(d_imgstack.shape[2]):
        for iwv in range(d_imgstack.shape[3]):
            _img = d_imgstack[:,:,iir,iwv]
            clip = np.percentile(_img,percentile)
            mask_imgstack[:,:,iir,iwv] = np.where(_img>clip,1,0)

    np.save(out_mask,mask_imgstack)