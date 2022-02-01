import numpy as np

from skimage import restoration
from skimage.exposure import equalize_adapthist

def deconvolve_img(img,psf=None):

    if psf is None:
        psf = getPSF()

    deconvolved_RL = restoration.richardson_lucy(img, psf, iterations=30)

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
    
    t = np.power(x/sig,2) + np.power(y/sig,2)

    
    t = -0.5*t/sig

    return np.exp(t)


if __name__=="__main__":
    size = 7
    x,y = np.mgrid[-int(size//2):int(size//2),
            -int(size//2):int(size//2)]

    print(x.shape)
    print(gaussianPSF(x,y,3).shape)