import sys
sys.path.append("../")
from utils.fileIO import Codebook,read_table
from utils.imgproc import warp_image
from utils.helper_functions import performDumbExtraction,findErrorBits
from .BaseReport import BaseReport
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.ndimage as ndi
import skimage
from sklearn.neighbors import NearestNeighbors

class DecodabilityReport(BaseReport):
    def __init__(self,deconvolved_img_stack,coord_info,codebook_file,data_org_file,fov,z):
        """_summary_

        Args:
            deconvolved_img_stack (_type_): _description_
            coord_info (_type_): _description_
            fov (_type_): _description_
            z (_type_): _description_
        """

        super().__init__(deconvolved_img_stack,coord_info)
        self.fov_name = fov
        self.z_name = z
        #Load the codebook file
        self.codebook = Codebook(codebook_file)

        #Load the dataorg file
        data_org = read_table(data_org_file).to_dict("records")

        #Warp the deconvolved image stack into (16,x,y)
        self.warped_imgs = warp_image(data_org,self.imgstack,16)

        #Pass into the perform the dumb extraction
        self.filtered_warped_images, self.bright_pixel_count, self.bit_map, \
            self.bit_vector_map, self.dist_map, self.barcode_map, self.error_bit_map, \
                self.detection_number_map,self.codebook_bits = performDumbExtraction(self.warped_imgs, self.codebook,clip_thresh=98.5)
    

    '''
    ====MAKE REPORTS=====
    '''

    def make_bit_imgs(self):
        #make 3,4,5 bit images
        image_blank = np.zeros((self.warped_imgs.shape[1],self.warped_imgs.shape[2]))

        bc3_img = np.multiply(np.reshape(self.barcode_map[3],image_blank.shape)+1,np.reshape(self.dist_map[3]==1,image_blank.shape))
        bc4_img = np.multiply(np.reshape(self.barcode_map[4],image_blank.shape)+1,np.reshape(self.dist_map[4]==0,image_blank.shape))
        bc5_img = np.multiply(np.reshape(self.barcode_map[5],image_blank.shape)+1,np.reshape(self.dist_map[5]==1,image_blank.shape))

        bc3_img = np.where(bc3_img>0,1,0)

        bc4_img = np.where(bc4_img>0,1,0)

        bc5_img = np.where(bc5_img>0,1,0)

        f,ax = plt.subplots(nrows=1,ncols=3,sharex=True,sharey=True,figsize=(3*10,10))

        ax[0].imshow(bc3_img,cmap='gray',vmin=0,vmax=1,interpolation='none')
        ax[1].imshow(bc4_img,cmap='gray',vmin=0,vmax=1,interpolation='none')
        ax[2].imshow(bc5_img,cmap='gray',vmin=0,vmax=1,interpolation='none')

        plt.tight_layout()
        self.pdf.savefig()
        plt.close(f)

        # Set the colour images

        colour_img = np.zeros((bc5_img.shape[0],bc5_img.shape[1],3))

        colour_img[:,:,0] = bc3_img
        colour_img[:,:,1] = bc4_img
        colour_img[:,:,2] = bc5_img

        f,ax = plt.subplots(figsize=(40,40))
        ax.imshow(colour_img,vmin=0,vmax=1,interpolation='none')
        ax.set_title('Red: 3-bit words; Green: 4-bit words; Blue: 5-bit words')
        plt.tight_layout()
        self.pdf.savefig()
        plt.close(f)



        #For each bit in the filtered warped image, spit out an RGB image 

        f,axs = plt.subplots(4,4,figsize=(40,40))
        axs = axs.flatten()
        window_half_width = 50

        r,c = 1200,800

        tl = [
            int(r-window_half_width),
            int(c-window_half_width)
        ]


        br = [
            int(r+window_half_width),
            int(c+window_half_width)
        ]

        fwi = np.moveaxis(self.filtered_warped_images, 0, -1)
        wi =  np.moveaxis(self.warped_imgs, 0, -1)
        masked3 = np.multiply(fwi,np.reshape(self.dist_map[3]==1,image_blank.shape)[..., None])

        masked4 = np.multiply(fwi,np.reshape(self.dist_map[4]==0,image_blank.shape)[..., None])

        masked5 = np.multiply(fwi,np.reshape(self.dist_map[5]==1,image_blank.shape)[..., None])

        cropped_mask3 = masked3[tl[0]:br[0]+1,tl[1]:br[1]+1,:] 
        cropped_mask4 = masked4[tl[0]:br[0]+1,tl[1]:br[1]+1,:] 
        cropped_mask5 = masked5[tl[0]:br[0]+1,tl[1]:br[1]+1,:] 

        assert np.any(cropped_mask3==1), 'This is empty'

        mpl.rcParams["image.interpolation"] = 'none'
        for iax,ax in enumerate(axs):
            colour_image = np.stack((cropped_mask3[:,:,iax],
                                    cropped_mask4[:,:,iax],
                                    cropped_mask5[:,:,iax]),
                                    axis=2
                                )

            ax.imshow(colour_image*255,vmin=0,vmax=1)
            ax.set_title(f'Bit: {iax}')
        plt.suptitle('Red: 3-bit words; Green: 4-bit words; Blue: 5-bit words')
        plt.tight_layout()
        self.pdf.savefig()
        plt.close(f)

        f,axs = plt.subplots(4,4,figsize=(40,40))
        axs = axs.flatten()

        masked3 = np.multiply(fwi,np.reshape(self.dist_map[3]==1,image_blank.shape)[..., None])

        masked4 = np.multiply(fwi,np.reshape(self.dist_map[4]==0,image_blank.shape)[..., None])

        masked5 = np.multiply(fwi,np.reshape(self.dist_map[5]==1,image_blank.shape)[..., None])

        cropped_mask3 = masked3[tl[0]:br[0]+1,tl[1]:br[1]+1,:] 
        cropped_mask4 = masked4[tl[0]:br[0]+1,tl[1]:br[1]+1,:] 
        cropped_mask5 = masked5[tl[0]:br[0]+1,tl[1]:br[1]+1,:] 

        row3,col3,z3= np.where(cropped_mask3==1)

        row4,col4,z4= np.where(cropped_mask4==1)

        row5,col5,z5= np.where(cropped_mask5==1)


        mpl.rcParams["image.interpolation"] = 'none'
        for iax,ax in enumerate(axs):
            crop = wi[tl[0]:br[0]+1,tl[1]:br[1]+1,iax]
            crap_max = ndi.maximum_filter(crop, size=2, mode='constant')
            peak = skimage.feature.peak_local_max(crap_max,min_distance=2)
            # print(peak)
            ax.imshow(crop,cmap='gray')
            ax.scatter(col3,row3,s=15,color='red',marker='*')
            ax.scatter(col4,row4,s=15,color='green',marker='*')
            ax.scatter(col5,row5,s=15,color='blue',marker='*')
            ax.scatter(peak[:,1],peak[:,0],s=10,color='yellow',marker='*')
            ax.set_title(f'Bit: {iax}')
        plt.suptitle('')
        plt.tight_layout()
        self.pdf.savefig()
        plt.close(f)
        
        f,axs = plt.subplots(4,4,figsize=(40,40))
        axs = axs.flatten()
        for iax,ax in enumerate(axs):
            crop = fwi[tl[0]:br[0]+1,tl[1]:br[1]+1,iax]
            crap_max = ndi.maximum_filter(crop, size=2, mode='constant')
            peak = skimage.feature.peak_local_max(crap_max,min_distance=2)
            # print(peak)
            ax.imshow(crop,cmap='gray')
            ax.scatter(col3,row3,s=15,color='red',marker='*')
            ax.scatter(col4,row4,s=15,color='green',marker='*')
            ax.scatter(col5,row5,s=15,color='blue',marker='*')
            ax.scatter(peak[:,1],peak[:,0],s=10,color='yellow',marker='*')
            ax.set_title(f'Bit: {iax}')
        plt.suptitle('')
        plt.tight_layout()
        self.pdf.savefig()
        plt.close(f)
                

    def make_decodability_hists(self):

        f,ax = plt.subplots(figsize=(10,10))
        plt.hist(self.bright_pixel_count.ravel(),bins=[i for i in range(17)],)
        ax.set_xlim([1,17])
        ax.set_ylim([0,0.6e6])
        ax.vlines(x=[3,6],ymin=0,ymax=ax.get_ylim()[1],colors=['r','r'])
        ax.set_xticks(ticks=[i+0.5 for i in range(17)],labels=[i for i in range(17)])
        ax.set_title('Number of bright pixels stacked')
        plt.tight_layout()
        self.pdf.savefig()
        plt.close(f)
        codebook_bits = np.array(self.codebook.barcode_arrays)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree",p=1).fit(codebook_bits)

        bright_pix_3bcs = np.reshape(self.bit_map[3],(16,-1)).T
        bright_pix_4bcs = np.reshape(self.bit_map[4],(16,-1)).T
        bright_pix_5bcs = np.reshape(self.bit_map[5],(16,-1)).T

        bright_pix_3bcs = bright_pix_3bcs[bright_pix_3bcs.sum(axis=1)>0,:]
        bright_pix_4bcs = bright_pix_4bcs[bright_pix_4bcs.sum(axis=1)>0,:]
        bright_pix_5bcs = bright_pix_5bcs[bright_pix_5bcs.sum(axis=1)>0,:]

        # Look at distances for 3 pixel barcodes
        dist3, barcode3_ids = nbrs.kneighbors(bright_pix_3bcs)
        dist3_fraction = (dist3==1).sum() /(dist3>0).sum() 
        f,ax = plt.subplots(figsize=(10,10))
        plt.hist(dist3.ravel(),bins=[i for i in range(17)])
        plt.vlines(x=[1,2],ymin=0,ymax=ax.get_ylim()[1],colors=['r','r'])
        ax.set_xticks(ticks=[i+0.5 for i in range(17)],labels=[i for i in range(17)])
        ax.set_xlabel('Manhattan Distance away from entry in codebook')
        ax.set_ylabel('Number of Pixels')
        plt.title('3 Bright Pixel Barcodes: Manhattan Distance \n Detection fraction: {:.02f}'.format(dist3_fraction))
        plt.tight_layout()
        self.pdf.savefig()
        plt.close(f)
        # Look at distances for 4 pixel barcodes
        dist4, barcode4_ids = nbrs.kneighbors(bright_pix_4bcs)
        dist4_fraction = (dist4==0).sum() /(dist4>-1).sum() 
        f,ax = plt.subplots(figsize=(10,10))
        plt.hist(dist4.ravel(),bins=[i for i in range(17)])
        plt.vlines(x=[0,1],ymin=0,ymax=ax.get_ylim()[1],colors=['r','r'])
        ax.set_xticks(ticks=[i+0.5 for i in range(17)],labels=[i for i in range(17)])
        ax.set_xlabel('Manhattan Distance away from entry in codebook')
        ax.set_ylabel('Number of Pixels')
        plt.title('4 Bright Pixel Barcodes: Manhattan Distance \n Detection fraction: {:.02f}'.format(dist4_fraction))
        plt.tight_layout()
        self.pdf.savefig()
        plt.close(f)


        # Look at distances for 5 pixel barcodes
        dist5, barcode5_ids = nbrs.kneighbors(bright_pix_5bcs)
        dist5_fraction = (dist5==1).sum() /(dist5>-1).sum() 
        f,ax = plt.subplots(figsize=(10,10))
        plt.hist(dist5.ravel(),bins=[i for i in range(17)])
        plt.vlines(x=[1,2],ymin=0,ymax=ax.get_ylim()[1],colors=['r','r'])
        ax.set_xticks(ticks=[i+0.5 for i in range(17)],labels=[i for i in range(17)])
        ax.set_xlabel('Manhattan Distance away from entry in codebook')
        ax.set_ylabel('Number of Pixels')
        plt.title('5 Bright Pixel Barcodes: Manhattan Distance  \n Detection fraction: {:.02f}'.format(dist5_fraction))
        plt.tight_layout()
        self.pdf.savefig()
        plt.close(f)



        '''
        Find out what is the bit distribution of the spots that had 3-5 bits, but didnt match the codebook
        '''

        unique_dist3 = np.unique(dist3)
        dist3_barcode_hist = np.zeros((16,16))
        for ud3 in unique_dist3:
            dist3_barcode_hist[int(ud3),:] = np.sum(bright_pix_3bcs[np.squeeze(dist3==ud3),:],axis=0)
            dist3_barcode_hist[int(ud3),:] =np.nan_to_num(dist3_barcode_hist[int(ud3),:]/ dist3_barcode_hist[int(ud3),:].sum())

        unique_dist4 = np.unique(dist4)
        dist4_barcode_hist = np.zeros((16,16))
        for ud4 in unique_dist4:
            dist4_barcode_hist[int(ud4),:] = np.sum(bright_pix_4bcs[np.squeeze(dist4==ud4),:],axis=0)
            dist4_barcode_hist[int(ud4),:] = np.nan_to_num(dist4_barcode_hist[int(ud4),:]/ dist4_barcode_hist[int(ud4),:].sum())

        unique_dist5 = np.unique(dist5)
        dist5_barcode_hist = np.zeros((16,16))
        for ud5 in unique_dist5:
            dist5_barcode_hist[int(ud5),:] = np.sum(bright_pix_5bcs[np.squeeze(dist5==ud5),:],axis=0)
            dist5_barcode_hist[int(ud5),:] =np.nan_to_num(dist5_barcode_hist[int(ud5),:]/ dist5_barcode_hist[int(ud5),:].sum())


        f,ax = plt.subplots(figsize=(10,10))
        plt.imshow(dist3_barcode_hist)
        for ud in range(dist3_barcode_hist.shape[0]):
            for bp in range(dist3_barcode_hist.shape[0]):
                text = ax.text(bp, ud, f'{dist3_barcode_hist[ud, bp]*100:.2f}%', ha="center", va="center", color="w")
        ax.set_ylabel('Manhattan Distance away from entry in codebook')
        ax.set_xlabel('Bit Number')
        plt.title('3 Bright Pixel bit distribution')
        plt.tight_layout()
        self.pdf.savefig()
        plt.close(f)

        f,ax = plt.subplots(figsize=(10,10))
        plt.imshow(dist4_barcode_hist)
        for ud in range(dist4_barcode_hist.shape[0]):
            for bp in range(dist4_barcode_hist.shape[0]):
                text = ax.text(bp, ud, f'{dist4_barcode_hist[ud, bp]*100:.2f}%', ha="center", va="center", color="w")
        ax.set_ylabel('Manhattan Distance away from entry in codebook')
        ax.set_xlabel('Bit Number')
        plt.title('4 Bright Pixel bit distribution')
        plt.tight_layout()
        self.pdf.savefig()
        plt.close(f)


        f,ax = plt.subplots(figsize=(10,10))
        plt.imshow(dist5_barcode_hist)
        for ud in range(dist5_barcode_hist.shape[0]):
            for bp in range(dist5_barcode_hist.shape[0]):
                text = ax.text(bp, ud, f'{dist5_barcode_hist[ud, bp]*100:.2f}%', ha="center", va="center", color="w")
        ax.set_ylabel('Manhattan Distance away from entry in codebook')
        ax.set_xlabel('Bit Number')
        plt.title('5 Bright Pixel bit distribution')
        plt.tight_layout()
        self.pdf.savefig()
        plt.close(f)

        errorBit3 = findErrorBits(bright_pix_3bcs,self.codebook_bits,dist3,barcode3_ids,filter_val=1)

        errorBit4 = findErrorBits(bright_pix_4bcs,self.codebook_bits,dist4,barcode4_ids,filter_val=0)

        errorBit5 = findErrorBits(bright_pix_5bcs,self.codebook_bits,dist5,barcode5_ids,filter_val=1)


        f,ax = plt.subplots(figsize=(10,10))
        ax.hist(errorBit3.ravel(),bins = [i for i in range(17)])
        plt.tight_layout()
        self.pdf.savefig()
        plt.close(f)


        f,ax = plt.subplots(figsize=(10,10))
        ax.hist(errorBit4.ravel(),bins = [i for i in range(17)])
        plt.tight_layout()
        self.pdf.savefig()
        plt.close(f)


        f,ax = plt.subplots(figsize=(10,10))
        ax.hist(errorBit5.ravel(),bins = [i for i in range(17)])
        plt.tight_layout()
        self.pdf.savefig()
        plt.close(f)

        # # Save the output of detected barcodes
        # number_report = [f"Number of 3bit Detections: {dist3[dist3==1].sum()}\n",
        # f"Number of 4bit Detections: {(1+dist4[dist4==0]).sum()}\n",
        # f"Number of 5bit Detections: {dist5[dist5==1].sum()}"]

        # with open(f'./{plot_folder}/detection_stats.txt','w') as f:
        #     f.writelines(number_report)


        # make image that show the spatial distribution of the 3-5 bit images

        bright_pix_3bcs = np.reshape(self.bit_map[3],(16,-1)).T
        bright_pix_4bcs = np.reshape(self.bit_map[4],(16,-1)).T
        bright_pix_5bcs = np.reshape(self.bit_map[5],(16,-1)).T
        dist3, barcode3_ids = nbrs.kneighbors(bright_pix_3bcs)
        dist4, barcode4_ids = nbrs.kneighbors(bright_pix_4bcs)
        dist5, barcode5_ids = nbrs.kneighbors(bright_pix_5bcs)


        image_blank = np.zeros((self.filtered_warped_images.shape[1],self.filtered_warped_images.shape[2]))

        bc3_img = np.multiply(np.reshape(barcode3_ids,image_blank.shape)+1,np.reshape(dist3,image_blank.shape)==1)>0
        bc4_img = np.multiply(np.reshape(barcode4_ids,image_blank.shape)+1,np.reshape(dist4,image_blank.shape)==0)>0
        bc5_img = np.multiply(np.reshape(barcode5_ids,image_blank.shape)+1,np.reshape(dist5,image_blank.shape)==1)>0


        f,ax = plt.subplots(figsize=(10,10))
        ax.imshow(bc3_img,cmap='gray')
        plt.tight_layout()
        self.pdf.savefig()
        plt.close(f)

        f,ax = plt.subplots(figsize=(10,10))
        ax.imshow(bc4_img,cmap='gray')
        plt.tight_layout()
        self.pdf.savefig()
        plt.close(f)

        f,ax = plt.subplots(figsize=(10,10))
        ax.imshow(bc5_img,cmap='gray')
        plt.tight_layout()
        self.pdf.savefig()
        plt.close(f)


if __name__=="__main__":
    print("test")