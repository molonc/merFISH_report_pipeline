import numpy as np

from skimage import restoration
from skimage.exposure import equalize_adapthist


# Deconvolute Images
def _deconvolute(raw_image):
    """ Deconvolutes image to sharpen features with filtering and richardson lucy restoration
    Args:
        raw_image: 2D image
        dtype: the dtype of the raw_image being passed in
    
    Returns:
        img: filtered and restored image. This image maintains the original dtype
    """
    img = raw_image.astype("uint16")
    # High pass filtering
    filt = (
        skimage.filters.gaussian(
            img, sigma=3, mode="nearest", truncate=2, preserve_range=True
        )
        .round()
        .astype("uint16")
    )
    img = np.subtract(img, filt, where=img > filt, out=np.zeros_like(img))

    # Point spread function deconvolution
    psf = _gaussian_kernel((10, 10), 2)

    np.where(img == np.nan, 0, img)
    img = skimage.restoration.richardson_lucy(img, psf, iterations=30, clip=False)

    # Low pass filtering
    img = skimage.filters.gaussian(
        img,
        sigma=1,
        output=None,
        cval=0,
        multichannel=False,
        preserve_range=True,
        truncate=4.0,
    )
    return img.astype(dtype=float)


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



