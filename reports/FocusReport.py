from .BaseReport import BaseReport

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scipy.ndimage as ndi

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


