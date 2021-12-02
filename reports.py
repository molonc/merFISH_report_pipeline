import skimage.io as skio
import scipy.ndimage as ndi
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
    
class BaseReport():
    # Base class for all the reports
    def __init__(self,imgstack_file,coord_info):
        
        self.imgstack = np.load(imgstack_file)
        a_file = open(coord_info, "r")
        self.coords = json.load(a_file)

        self.pdf = None
    #PDF controls--------------------------------------------
    def set_pdf(self,pdf)-> None:
        self.pdf = pdf
    
    def get_pdf(self):
        return self.pdf

    def isPdf(self)->bool:
        if self.pdf is None:
            return False
        return True
    
    def closePdf(self)->None:
        if self.isPdf():
            self.pdf.close()
            self.pdf = None

class BrightnessReport(BaseReport):
    def __init__(self,imgstack_file,coord_info,fov,fovs):
        super().__init__(imgstack_file,coord_info)
        
        self.fov_name = fov
        self.imgstack = self.imgstack[:,:,:,:,0,:] # Each image stack is only 1 FOV
    #Reports----------------------------------------------
    def brightness_through_z(self):
        #Take the image stack and do a max projection from all the pixels through z
        mip_z_stack = self.imgstack.max(axis=4)

        largest = mip_z_stack.max()
        f,ax = plt.subplots(nrows=len(self.coords['irs']),ncols=len(self.coords['wvs']),sharex=True,sharey=True,figsize=(len(self.coords['wvs'])*4,len(self.coords['irs'])*4))
        if not isinstance(ax,np.ndarray):
            ax = np.array([ax])
        if len(ax.shape)==1:
            ax=ax[:,np.newaxis]
        plt.suptitle(f'FOV: {self.fov_name}')
        for iwv,wv in enumerate(self.coords['wvs']):
            for iir,ir in enumerate(self.coords['irs']):
                bin_num = int(largest//2)

                data = mip_z_stack[:,:,iwv,iir].flatten()

                ax[iir,iwv].hist(data,bins=bin_num,range=(0,largest),log=True,histtype='step')

                if iwv==0:
                    ax[iir,iwv].set_ylabel(f'ir:{ir}')
                if iir==0:
                    ax[iir,iwv].set_title(f'channel: {wv}nm')

        plt.tight_layout()
        self.pdf.savefig()
        plt.close(f)


    def brightness_through_z_on_images(self):

        mip_z_stack = self.imgstack.max(axis=4)

        max_wv_val = self.imgstack.max(axis=0).max(axis=0).max(axis=1).max(axis=1)
        val_range = max_wv_val*0.9
        largest = mip_z_stack.max()
        f,ax = plt.subplots(nrows=len(self.coords['irs']),ncols=len(self.coords['wvs']),sharex=True,sharey=True,figsize=(len(self.coords['wvs'])*4,len(self.coords['irs'])*4))
        plt.suptitle(f'FOV: {self.fov_name}')
        if not isinstance(ax,np.ndarray):
            ax = np.array([ax])
        if len(ax.shape)==1:
            ax=ax[:,np.newaxis]
        for iwv,wv in enumerate(self.coords['wvs']):
            for iir,ir in enumerate(self.coords['irs']):
                img = mip_z_stack[:,:,iwv,iir]
                bin_num = int(largest//2)
                ax[iir,iwv].imshow(img,vmax=val_range[iwv],cmap='gray')
                # Add histogram to the corner of the image
                axins = inset_axes(ax[iir,iwv], width="25%", height="25%", loc=4, borderpad=1)
                data = img.flatten()
                axins.hist(data,bins=bin_num,range=(0,largest),log=True,histtype='step')
                axins.tick_params(labelleft=False, labelbottom=False)
                if iwv==0:
                    ax[iir,iwv].set_ylabel(f'ir:{ir}')
                if iir==0:
                    ax[iir,iwv].set_title(f'channel: {wv}nm')

        plt.tight_layout()
        self.pdf.savefig()
        plt.close(f)

    def preview_images(self):
        mip_z_stack = self.imgstack.max(axis=4)
        max_wv_val = self.imgstack.max(axis=0).max(axis=0).max(axis=1).max(axis=1)
        val_range = max_wv_val*0.9#self.imgstack.max()*0.90

        f,ax = plt.subplots(nrows=len(self.coords['irs']),ncols=len(self.coords['wvs']),sharex=True,sharey=True,figsize=(len(self.coords['wvs'])*4,len(self.coords['irs'])*4))
        plt.suptitle(f'FOV: {self.fov_name}')
        if not isinstance(ax,np.ndarray):
            ax = np.array([ax])
        if len(ax.shape)==1:
            ax=ax[:,np.newaxis]
        for iwv,wv in enumerate(self.coords['wvs']):
            for iir,ir in enumerate(self.coords['irs']):
                img = mip_z_stack[:,:,iwv,iir]
                ax[iir,iwv].imshow(img,vmax=val_range[iwv],cmap='gray')
                if iwv==0:
                    ax[iir,iwv].set_ylabel(f'ir:{ir}')
                if iir==0:
                    ax[iir,iwv].set_title(f'channel: {wv}nm')

        plt.tight_layout()
        self.pdf.savefig()
        plt.close(f)



class FocusReport(BaseReport):
    def __init__(self,imgstack_file,coord_info,fov,fovs):
        super().__init__(imgstack_file,coord_info)
        
        self.fov_name = fov
        self.imgstack = self.imgstack[:,:,:,:,0,:]

        self.peak_idx = np.zeros((self.imgstack.shape[2],self.imgstack.shape[3]))
    
    def f_measure(self,img):
        filter = np.array([ [1],
                            [0],
                            [-1],
                            [0],
                            [0]])

        res = ndi.convolve(img,filter,mode='reflect')
        res = np.power(res,2)
        return res.sum()

    #Reports-----
    def f_measure_report(self):
        #Take the image stack and do a max projection from all the pixels through z
        f,ax = plt.subplots(nrows=len(self.coords['irs']),ncols=1,sharex=True,sharey=True,figsize=(len(self.coords['zs'])*1,15))
        if not isinstance(ax,np.ndarray):
            ax = np.array([ax])
        plt.suptitle(f'FOV: {self.fov_name}')
        for iir,ir in enumerate(self.coords['irs']):
            ax[iir].set_title(f'ir: {ir}')
            for iwv,wv in enumerate(self.coords['wvs']):
                output=[]
                for iz,z in enumerate(self.coords['zs']):
                    fnum = self.f_measure(self.imgstack[:,:,iwv,iir,iz])
                    output.append(fnum)
                
                ax[iir].plot(output,color=base_colors[iwv])
                _output = np.array(output)
                peak = _output.argmax()
                self.peak_idx[iwv,iir]=peak
                ax[iir].vlines(peak,0,_output[peak],colors=base_colors[iwv],linestyles='dashed')
                ax[iir].set_yscale('log')
            ax[iir].legend(self.coords['wvs'])
        
        plt.tight_layout()
        self.pdf.savefig()
        plt.close(f)

# Brightness report worker-------
def generate_brightness_reports(image_stack_file,coord_info,out_file,fov,fovs):
    #This was necessary to prototype on OSX...consider removing in the future after testing
    import multiprocessing
    job = multiprocessing.Process(target=brightness_worker,args=(image_stack_file,coord_info,out_file,fov,fovs))
    job.start()

    #brightness_worker(image_stack_file,coord_info,out_file,fov,fovs)

def brightness_worker(image_stack_file,coord_info,out_file,fov,fovs):
    br = BrightnessReport(image_stack_file,coord_info,fov,fovs)
    br.set_pdf(PdfPages(filename=out_file))
    
    br.preview_images()
    br.brightness_through_z()
    br.brightness_through_z_on_images()
    br.closePdf()


# Focus report worker-------
def generate_focus_reports(image_stack_file,coord_info,out_file,fov,fovs):
    #This was necessary to prototype on OSX...consider removing in the future after testing
    import multiprocessing
    job = multiprocessing.Process(target=focus_worker,args=(image_stack_file,coord_info,out_file,fov,fovs))
    job.start()
    # focus_worker(image_stack_file,coord_info,out_file,fov,fovs)


def focus_worker(image_stack_file,coord_info,out_file,fov,fovs):
    fr = FocusReport(image_stack_file,coord_info,fov,fovs)
    fr.set_pdf(PdfPages(filename=out_file))
    fr.f_measure_report()
    fr.closePdf()
