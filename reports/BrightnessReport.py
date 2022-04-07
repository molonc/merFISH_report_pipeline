from .BaseReport import BaseReport

from utils import imgproc

import skimage.exposure as ske
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes




class BrightnessReport(BaseReport):
    def __init__(self,imgstack_file,coord_info,fov,z):
        super().__init__(imgstack_file,coord_info)
        
        self.fov_name = fov
        self.z_name = z
        self.imgstack = self.imgstack # (y,x,wvs,irs)
        self.contrast_tape = np.zeros((self.imgstack.shape[2],
                                    self.imgstack.shape[3]))




    #Helper----------------------------------------------
    def calc_HS_metric(self,img):
        '''
        https://ieeexplore.ieee.org/document/6108900
        '''
        vals = np.percentile(img.ravel(),[75,25])
        max_val = np.max(img)
        min_val = np.min(img)

        return (vals[0]-vals[1])/(max_val-min_val)
    def calc_HF_metric(self,img):
        '''
        https://ieeexplore.ieee.org/document/6108900
        '''

        hist,_ = np.histogram(img.flatten(),bins = int(img.max()//2),range=(0,img.max()))

        return np.power(np.prod(hist),1/len(hist)) / hist.sum() * len(hist)

        
    def ski_is_low_contrast(self,img,fraction_threshold = 0.25):
        return ske.is_low_contrast(img,fraction_threshold=fraction_threshold)

    def contrast_test(self,img,threshold=0.25,method='ski'):
        if method=='ski':
            return self.ski_is_low_contrast(img,threshold)
        elif method =='HS':
            res = self.calc_HS_metric(img)
            return res>threshold
        elif method =='HF':
            res = self.calc_HF_metric(img)
            return res>threshold

        



    #Reports----------------------------------------------
    
    def preview_images(self):
        max_wv_val = self.imgstack.max(axis=0).max(axis=0).max(axis=1)
        val_range = max_wv_val*0.9#self.imgstack.max()*0.90
        f,ax = plt.subplots(nrows=len(self.coords['irs']),ncols=len(self.coords['wvs']),sharex=True,sharey=True,figsize=(len(self.coords['wvs'])*4,len(self.coords['irs'])*4))
        plt.suptitle(f'FOV: {self.fov_name}; Z: {self.z_name}')
        if not isinstance(ax,np.ndarray):
            ax = np.array([ax])
        if len(ax.shape)==1:
            if len(self.coords['irs'])==1:
                ax=ax[np.newaxis,:]
            elif len(self.coords['wvs'])==1:
                ax=ax[:,np.newaxis]
        for iwv,wv in enumerate(self.coords['wvs']):
            for iir,ir in enumerate(self.coords['irs']):
                img = self.imgstack[:,:,iwv,iir]
                ax[iir,iwv].imshow(img,vmax=val_range[iwv],cmap='gray')
                if iwv==0:
                    ax[iir,iwv].set_ylabel(f'ir:{ir}')
                if iir==0:
                    ax[iir,iwv].set_title(f'channel: {wv}nm')

        plt.tight_layout()
        self.pdf.savefig()
        plt.close(f)


    def brightness_infov_z(self):
        largest = self.imgstack.max()
        f,ax = plt.subplots(nrows=len(self.coords['irs']),ncols=len(self.coords['wvs']),sharex=True,sharey=True,figsize=(len(self.coords['wvs'])*4,len(self.coords['irs'])*4))
        if not isinstance(ax,np.ndarray):
            ax = np.array([ax])
        if len(ax.shape)==1:
            if len(self.coords['irs'])==1:
                ax=ax[np.newaxis,:]
            elif len(self.coords['wvs'])==1:
                ax=ax[:,np.newaxis]
        plt.suptitle(f'FOV: {self.fov_name}; Z: {self.z_name}')
        
        for iwv,wv in enumerate(self.coords['wvs']):
            for iir,ir in enumerate(self.coords['irs']):
                bin_num = int(largest//2)

                data = self.imgstack[:,:,iwv,iir]
                self.contrast_tape[iwv,iir] = self.calc_HS_metric(data)
                flat_data = data.ravel()

                ax[iir,iwv].hist(flat_data,bins=bin_num,range=(0,largest),log=True,histtype='step')
                ax[iir,iwv].text(0.5, 0.5, f'HS:{self.contrast_tape[iwv,iir]}',
                                ha="center", va="center",
                                transform=ax[iir,iwv].transAxes)
                if iwv==0:
                    ax[iir,iwv].set_ylabel(f'ir:{ir}')
                if iir==0:
                    ax[iir,iwv].set_title(f'channel: {wv}nm')

        plt.tight_layout()
        self.pdf.savefig()
        plt.close(f)
    
    def brightness_on_images(self):


        max_wv_val = self.imgstack.max(axis=0).max(axis=0).max(axis=1)
        val_range = max_wv_val*0.9
        largest = self.imgstack.max()
        f,ax = plt.subplots(nrows=len(self.coords['irs']),ncols=len(self.coords['wvs']),sharex=True,sharey=True,figsize=(len(self.coords['wvs'])*4,len(self.coords['irs'])*4))
        plt.suptitle(f'FOV: {self.fov_name}; Z: {self.z_name}')
        if not isinstance(ax,np.ndarray):
            ax = np.array([ax])
        if len(ax.shape)==1:
            if len(self.coords['irs'])==1:
                ax=ax[np.newaxis,:]
            elif len(self.coords['wvs'])==1:
                ax=ax[:,np.newaxis]
        for iwv,wv in enumerate(self.coords['wvs']):
            for iir,ir in enumerate(self.coords['irs']):
                img = self.imgstack[:,:,iwv,iir]
                bin_num = int(largest//2)
                ax[iir,iwv].imshow(img,vmax=val_range[iwv],cmap='gray')
                # Add histogram to the corner of the image
                axins = inset_axes(ax[iir,iwv], width="25%", height="25%", loc=4, borderpad=1)
                data = img.ravel()
                axins.hist(data,bins=bin_num,range=(0,largest),log=True,histtype='step')
                axins.tick_params(labelleft=False, labelbottom=False)
                if iwv==0:
                    ax[iir,iwv].set_ylabel(f'ir:{ir}')
                if iir==0:
                    ax[iir,iwv].set_title(f'channel: {wv}nm')

        plt.tight_layout()
        self.pdf.savefig()
        plt.close(f)
    
    def contrast_heatmap(self):

        f,ax = plt.subplots()

        ims = ax.imshow(self.contrast_tape)
        ax.set_yticks(np.arange(len(self.coords['wvs'])), self.coords['wvs'])
        ax.set_xticks(np.arange(len(self.coords['irs'])), self.coords['irs'])
        ax.set_ylabel("Wavelength (nm)")
        ax.set_xlabel("Imaging Round")

        ax.set_title(f'FOV: {self.fov_name}; Z: {self.z_name}')
        plt.colorbar(ims)
        plt.tight_layout()
        self.pdf.savefig()
        plt.close(f)
        
    def _brightness_through_z(self):
        """This is deprecated.
        """
        #Take the image stack and do a max projection from all the pixels through z
        mip_z_stack = self.imgstack.max(axis=4)

        largest = mip_z_stack.max()
        f,ax = plt.subplots(nrows=len(self.coords['irs']),ncols=len(self.coords['wvs']),sharex=True,sharey=True,figsize=(len(self.coords['wvs'])*4,len(self.coords['irs'])*4))
        if not isinstance(ax,np.ndarray):
            ax = np.array([ax])
        if len(ax.shape)==1:
            if len(self.coords['irs'])==1:
                ax=ax[np.newaxis,:]
            elif len(self.coords['wvs'])==1:
                ax=ax[:,np.newaxis]
        plt.suptitle(f'FOV: {self.fov_name}')
        for iwv,wv in enumerate(self.coords['wvs']):
            for iir,ir in enumerate(self.coords['irs']):
                bin_num = int(largest//2)

                data = mip_z_stack[:,:,iwv,iir]
                self.contrast_tape[iwv,iir] = self.calc_HS_metric(data)
                flat_data = data.ravel()

                ax[iir,iwv].hist(flat_data,bins=bin_num,range=(0,largest),log=True,histtype='step')
                ax[iir,iwv].text(0.5, 0.5, f'HS:{self.contrast_tape[iwv,iir]}',
                                ha="center", va="center",
                                transform=ax[iir,iwv].transAxes)
                if iwv==0:
                    ax[iir,iwv].set_ylabel(f'ir:{ir}')
                if iir==0:
                    ax[iir,iwv].set_title(f'channel: {wv}nm')

        plt.tight_layout()
        self.pdf.savefig()
        plt.close(f)


    def _brightness_through_z_on_images(self):
        """This is Deprecated.
        """
        mip_z_stack = self.imgstack.max(axis=4)

        max_wv_val = self.imgstack.max(axis=0).max(axis=0).max(axis=1).max(axis=1)
        val_range = max_wv_val*0.9
        largest = mip_z_stack.max()
        f,ax = plt.subplots(nrows=len(self.coords['irs']),ncols=len(self.coords['wvs']),sharex=True,sharey=True,figsize=(len(self.coords['wvs'])*4,len(self.coords['irs'])*4))
        plt.suptitle(f'FOV: {self.fov_name}')
        if not isinstance(ax,np.ndarray):
            ax = np.array([ax])
        if len(ax.shape)==1:
            if len(self.coords['irs'])==1:
                ax=ax[np.newaxis,:]
            elif len(self.coords['wvs'])==1:
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


        
    def _preview_images(self):
        """This is deprecated.
        """
        mip_z_stack = self.imgstack.max(axis=4)
        max_wv_val = self.imgstack.max(axis=0).max(axis=0).max(axis=1).max(axis=1)
        val_range = max_wv_val*0.9#self.imgstack.max()*0.90

        f,ax = plt.subplots(nrows=len(self.coords['irs']),ncols=len(self.coords['wvs']),sharex=True,sharey=True,figsize=(len(self.coords['wvs'])*4,len(self.coords['irs'])*4))
        plt.suptitle(f'FOV: {self.fov_name}')
        if not isinstance(ax,np.ndarray):
            ax = np.array([ax])
        if len(ax.shape)==1:
            if len(self.coords['irs'])==1:
                ax=ax[np.newaxis,:]
            elif len(self.coords['wvs'])==1:
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
