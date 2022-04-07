from reports.BrightnessReport import BrightnessReport
import numpy as np


class MaskedBrightnessReport(BrightnessReport):
    def __init__(self,imgstack_file,masked_images,coord_info,fov,z):
        super().__init__(imgstack_file,coord_info,fov,z)
        self.mask_imgstack = self._read_imgstack(masked_images)# (y,x,wvs,irs)

        self.imgstack = np.multiply(self.imgstack,self.mask_imgstack)
    