import numpy as np
from functools import partial
from skimage import restoration,filters
import skimage.io as skio
from skimage.exposure import equalize_adapthist
import skimage
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

def warp_image(data_org_path, image_paths, output_path, config):
    r"""Combines imaging rounds of a single fov into a stack after deconvolution
    Args:
        data_org_path: path to data organaization
        image_paths: array of image paths for each imaging round (tiff or numpy)
        output_path: output path for image stack (numpy)
        config: pickled config object

    Returns:
        No return. Saves image stack as numpy array to output_path

    Raises:
        ValueError: if unexpected colour values are detected

    Example of how to call warp_images:
        ims = [
            r'C:\Users\asmith\Documents\vancouver-python\sandbox\ims\merFISH_merged_00_000.tif',
            r'C:\Users\asmith\Documents\vancouver-python\sandbox\ims\merFISH_merged_01_000.tif',
            r'C:\Users\asmith\Documents\vancouver-python\sandbox\ims\merFISH_merged_02_000.tif',
            r'C:\Users\asmith\Documents\vancouver-python\sandbox\ims\merFISH_merged_03_000.tif',
            r'C:\Users\asmith\Documents\vancouver-python\sandbox\ims\merFISH_merged_04_000.tif',
            r'C:\Users\asmith\Documents\vancouver-python\sandbox\ims\merFISH_merged_05_000.tif',
            r'C:\Users\asmith\Documents\vancouver-python\sandbox\ims\merFISH_merged_06_000.tif',
            r'C:\Users\asmith\Documents\vancouver-python\sandbox\ims\merFISH_merged_07_000.tif'
        ]
        data_org_p = r'C:\Users\asmith\Documents\vancouver-python\sandbox\ims\data_organization.csv'
        out_p = r'C:\Users\asmith\Documents\vancouver-python\sandbox\output\warped_updated.npy'
        config = {'ir':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]}
        warp_image(data_org_p, ims, out_p, config)
    """
    
    data_org = load_data_organization(data_org_path)
    warped_image_stack = []
    for bit_num in range(config["bit_num"]):
        data_org_row = data_org[bit_num]
        frame = int(data_org_row["frame"]) - 1

        # Reads the image for a single imaging round
        image_path = image_paths[int(data_org_row["imagingRound"])]
        im = skio.imread(image_path)
        shape = im.shape
        im = np.transpose(im, [len(shape) - 1] + list(range(len(shape) - 1)))
        warped_image_stack.append(im[frame])

    # pool = multiprocessing.get_context("forkserver").Pool()
    warped_image_stack = map(
        partial(_deconvolute), warped_image_stack
        # partial(preprocess, dtype=config["dtype"]), warped_image_stack
    )
    np.save(output_path, np.asarray(list(warped_image_stack)))

# Deconvolute Images
def _deconvolute(image_stack,out_file):
    """ Deconvolutes image to sharpen features with filtering and richardson lucy restoration
    Args:
        raw_image: 2D image
        dtype: the dtype of the raw_image being passed in
    
    Returns:
        img: filtered and restored image. This image maintains the original dtype
    """

    _imgstack = np.load(image_stack) #(y,x,ir,wv)
    d_imgstack = np.zeros_like(_imgstack)
    for iir in range(_imgstack.shape[2]):
        for iwv in range(_imgstack.shape[3]):
            raw_image = _imgstack[:,:,iir,iwv]

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
            img = restoration.richardson_lucy(img, psf, iterations=30, clip=False)

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

            d_imgstack[:,:,iir,iwv] = img
    
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