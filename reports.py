from numpy.core.numeric import full
import skimage.io as skio
import skimage.exposure as ske
import scipy.ndimage as ndi
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import matplotlib.colors as mcolors

import pandas as pd

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




    #Helper----------------------------------------------
    def calc_HS_metric(self,img):
        '''
        https://ieeexplore.ieee.org/document/6108900
        '''
        vals = np.percentile(img,[0.75,0.25])
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
    def brightness_through_z(self):
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

    def preview_images(self):
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



class FocusReport(BaseReport):
    def __init__(self,imgstack_file,coord_info,fov,fovs,out_csv):
        super().__init__(imgstack_file,coord_info)
        
        self.fov_name = fov
        self.imgstack = self.imgstack[:,:,:,:,0,:]

        self.peak_idx = np.zeros((self.imgstack.shape[2],self.imgstack.shape[3]))
        self.out_csv = out_csv
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


        #create an empty matrix with dimensions of ir x wv x z (rows v cols v depth) 
        #Populate the matrix in the loop
        #Then imshow and add text

        focus_matrix = np.zeros((len(self.coords['irs']),
                                len(self.coords['wvs']),
                                len(self.coords['zs']))
                                )

        for iir,ir in enumerate(self.coords['irs']):
            for iwv,wv in enumerate(self.coords['wvs']):
                for iz,z in enumerate(self.coords['zs']):
                    focus_matrix[iir,iwv,iz] = self.f_measure(self.imgstack[:,:,iwv,iir,iz])

        viz_focus_matrix = np.argmax(focus_matrix,axis=2)

        f, ax = plt.subplots()
        im = ax.imshow(viz_focus_matrix)     

        ax.set_xticks(np.arange(len(self.coords['wvs'])), self.coords['wvs'])
        ax.set_yticks(np.arange(len(self.coords['irs'])), self.coords['irs'])
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Imaging Round")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for iir in range(len(self.coords['irs'])):
            for iwv in range(len(self.coords['wvs'])):
                text = ax.text(iwv, iir, viz_focus_matrix[iir, iwv], ha="center", va="center", color="w")


        # #Take the image stack and do a max projection from all the pixels through z
        # f,ax = plt.subplots(nrows=len(self.coords['irs']),ncols=1,sharex=True,sharey=True,figsize=(len(self.coords['zs'])*1,15))
        # if not isinstance(ax,np.ndarray):
        #     ax = np.array([ax])
        # plt.suptitle(f'FOV: {self.fov_name}')
        # for iir,ir in enumerate(self.coords['irs']):
        #     ax[iir].set_title(f'ir: {ir}')
        #     for iwv,wv in enumerate(self.coords['wvs']):
        #         output=[]
        #         for iz,z in enumerate(self.coords['zs']):
        #             fnum = self.f_measure(self.imgstack[:,:,iwv,iir,iz])
        #             output.append(fnum)
                
        #         ax[iir].plot(output,color=base_colors[iwv])
        #         _output = np.array(output)
        #         peak = _output.argmax()
        #         self.peak_idx[iwv,iir]=peak
        #         ax[iir].vlines(peak,0,_output[peak],colors=base_colors[iwv],linestyles='dashed')
        #         ax[iir].set_yscale('log')
        #     ax[iir].legend(self.coords['wvs'])
        plt.tight_layout()
        self.pdf.savefig()
        plt.close(f)

        #Save the pk idx to a csv information for each imaging round
        df = pd.DataFrame()
        for iwv,wv in enumerate(self.coords['wvs']):
            df[str(wv)] = viz_focus_matrix[:,iwv] # each imaging round is a row
            df["FOV"] = len(viz_focus_matrix[:,0])*[self.fov_name]
            df["IR"] = self.coords['irs']
        df.to_csv(self.out_csv)    

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
def generate_focus_reports(image_stack_file,coord_info,out_file,out_csv,fov,fovs):
    #This was necessary to prototype on OSX...consider removing in the future after testing
    import multiprocessing
    job = multiprocessing.Process(target=focus_worker,args=(image_stack_file,coord_info,out_file,out_csv,fov,fovs))
    job.start()
    # focus_worker(image_stack_file,coord_info,out_file,out_csv,fov,fovs)


def focus_worker(image_stack_file,coord_info,out_file,out_csv,fov,fovs):
    fr = FocusReport(image_stack_file,coord_info,fov,fovs,out_csv)
    fr.set_pdf(PdfPages(filename=out_file))
    fr.f_measure_report()
    fr.closePdf()

# Compile reports----------------

def compile_focus_report(file_list:list,output,irs,wvs):
    
    full_df = pd.DataFrame()
    for fl in file_list:
        test = pd.read_csv(fl)
        full_df = full_df.append(test,ignore_index=True)
    
    full_df.to_csv(output)
    if not isinstance(irs,list):
        irs = list(irs)
    report_pdf = PdfPages(filename = output)
    

    compiled_matrix = np.zeros((
        len(file_list), # length of fovs
        len(wvs),
        len(irs)
    ))

    for iir, ir in enumerate(irs):#for each ir
        
        for iwv, wv in enumerate(wvs):#for each wv
            data = full_df[["FOV",str(wv)]][(full_df["IR"]==int(ir))] #extract the FOVs...assume stuff is ordered
            
            data = data.to_numpy() # FOV x 2

            compiled_matrix[:,iwv,iir] = data[:,1] #this is the z

            #ax[iir].plot("FOV",str(wv),data=data)
    #x:FOV y:z spy:IR
    f,ax = plt.subplots(nrows = len(wvs),ncols = 1,sharex=True,sharey=True,figsize=(8,len(irs)*4))    
    
    if not isinstance(ax,np.ndarray):
        ax = np.array(ax)
    
    for iwv, wv in enumerate(wvs):#for each wv
        ax[iwv].imshow(compiled_matrix[:,iwv,:])
        ax[iwv].set_xticks(np.arange(len(irs)), irs)
        ax[iwv].set_yticks(np.arange(len(file_list)), np.arange(len(file_list)))
        ax[iwv].set_ylabel("FOVS ")
        ax[iwv].set_xlabel("Imaging Round")
        plt.setp(ax[iwv].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for iir in range(len(irs)):
            for ifl in range(len(file_list)):
                text = ax[iwv].text(iir, ifl, int(compiled_matrix[ifl,iwv, iir]), ha="center", va="center", color="w")
        ax[iwv].set_title(f"Wavelength: {wv} nm")

    # for iir, ir in enumerate(irs):#for each ir
        
    #     for iwv, wv in enumerate(wvs):#for each wv
    #         data = full_df[["FOV",str(wv)]][(full_df["IR"]==int(ir))] #extract the FOVs...assume stuff is ordered
            
    #         ax[iir].plot("FOV",str(wv),data=data)
            
    #     ax[iir].set_ylabel(f"IR:{iir}")
    #     ax[iir].legend(wvs)
    # ax[iir].set_xlabel('FOVS')
    # plt.suptitle("In focus Z v FOV")
    report_pdf.savefig()
    report_pdf.close()
