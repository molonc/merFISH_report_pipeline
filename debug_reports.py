import json
import os
import numpy as np
from utils import fileIO
import reports
import makereports
from pathlib import Path
import time
import re

coord_file = "config.json"
a_file = open(coord_file, "r")
config = json.load(a_file)


fov = '001'
z='02'

def download_azure():
    
    if config["isRemote"]=="N" or os.path.exists(os.path.join(config['results_path'],'data')):
        return None
    
    os.system(config["azure_command"].format(config["raw_data_path"],os.path.join(config['results_path'],'data')))
    
    return None

def check_params():

    # Check to see if there are any params that need to be found
    b_findz = len(config['z'])==0
    b_findfov = len(config['fov'])==0
    b_findir = len(config['ir'])==0
    b_findwv = len(config['channel'])==0


    # get all the images
    if config["isRemote"] == "Y":
        files = Path(os.path.join(config['results_path'],'data')).rglob('*.TIFF')
    else:
        files = Path(config['raw_data_path']).rglob('*.TIFF')
    files = list(map(str,files))
    # Find all channel, ir, fov, and z from the file names
    re_filter = r"(.*)(\d{3})(?=nm).*(\B\d{2})\D(\B\d{3})\D(\B\d{2}\b)"
    param_filter = re.compile(re_filter)
    results = list(map(param_filter.search,files))
    
    results =list(filter(lambda x: isinstance(x,re.Match), results))
    # Get the list of parameters if they are needed
    #   Group 0 is the full match, so each capture group is 1 indexed
    if b_findz:
        zs = list(map(lambda x: x.group(5),results))
    else:
        zs = config['z']

    if b_findir:
        irs = list(map(lambda x: x.group(3),results))
    else:
        irs = config['ir'] 

    if b_findfov:
        fovs = list(map(lambda x: x.group(4),results))
    else:
        fovs = config['fov']

    if b_findwv:
        wvs = list(map(lambda x: x.group(2),results))
    else:
        wvs = config['channel']

    if isinstance(results,list) and len(results)>0:
        full_raw_path = results[0].group(1)
    else:
        full_raw_path=''


    return sorted(list(set(zs)),key=int),sorted(list(set(fovs)),key=int),sorted(list(set(irs)),key=int),sorted(list(set(wvs)),key=int),full_raw_path


def check_dirs(files:list)->None:
    """Checks to see if the directories for the files in the list exist. If they dont, then make those directories

    Args:
        files (list[str]): the list of files whose directory paths to create. Needs to be the full, not relative paths
        
    """
    if not type(files) == list:
        d = os.path.dirname(files)
        if not os.path.isdir(d):
            # print(files+'files' + d +'Does not exist')
            os.makedirs(d)
    else:
        for f in files:

            d = os.path.dirname(f)
            if d != "" and not os.path.isdir(d):
                # print(f+'files' + d +'Does not exist')
                os.makedirs(d)

download_azure()
zs,fovs,irs,wvs,full_raw_path = check_params()

def create_image_stack():

    out_file = os.path.join(config['results_path'],f'imgstack_{fov}_{z}.npy')
    coord_file = os.path.join(config['results_path'],f'coord_{fov}_{z}.json')
    check_dirs(out_file)

    fileIO.create_image_stack(os.path.join(full_raw_path,config['raw_image_format']),fov,z,irs,wvs,out_file,coord_file)


def create_brightness_report():
    
    img_stack = os.path.join(config['results_path'],f'imgstack_{fov}_{z}.npy')
    coord_file = os.path.join(config['results_path'],f'coord_{fov}_{z}.json')
    
    out=os.path.join(config['results_path'],f'brightness_report_{fov}_{z}.pdf')
    check_dirs(out)
    
    makereports.generate_brightness_reports(img_stack,coord_file,out,fov,z)

def create_focus_report():
    
    img_stack = [os.path.join(config['results_path'],f'imgstack_{fov}_{z}.npy')]
    coord_file = [os.path.join(config['results_path'],f'coord_{fov}_{z}.json')]
    
    out=os.path.join(config['results_path'],f'focus_report_{fov}.pdf')
    out_csv = os.path.join(config['results_path'],f'focus_report_{fov}.csv')
    check_dirs(out)
    makereports.generate_focus_reports(img_stack,coord_file,out,out_csv,fov)


def compile_focus_reports():

    in_files = [f'focus_{fov}.csv']
    output = 'full_report_debug.csv'
    
    makereports.compile_focus_report(in_files,output=output,irs=irs,wvs=wvs)


if __name__=='__main__':
    start_time = time.time()

    sub_start_time = time.time()
    create_image_stack()
    sub_end_time = time.time()
    print(f'Image Stack: {sub_end_time-sub_start_time}')
    sub_start_time = time.time()
    create_brightness_report()
    sub_end_time = time.time()
    print(f'Brightness Report: {sub_end_time-sub_start_time}')

    sub_start_time = time.time()
    create_focus_report()
    sub_end_time = time.time()
    print(f'Focus Report: {sub_end_time-sub_start_time}')

    sub_start_time = time.time()
    compile_focus_reports()
    sub_end_time = time.time()
    print(f'Compiled Focus Report: {sub_end_time-sub_start_time}')    

    end_time = time.time()
    print(f'Total Time: {end_time-start_time}')