from reports import BrightnessReport, FocusReport
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import pandas as pd


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
    br.contrast_heatmap()
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


    report_pdf.savefig()
    report_pdf.close()
