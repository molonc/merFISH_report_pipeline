from pathlib import Path
import re
import os
import reports

configfile: "config.json"

def check_params():
    print('Checking Params')
    # Check to see if there are any params that need to be found
    b_findz = len(config['z'])==0
    b_findfov = len(config['fov'])==0
    b_findir = len(config['ir'])==0
    b_findwv = len(config['channel'])==0

    # if all the params are already specified, then just return those and save time
    if (not b_findfov) and (not b_findir) and (not b_findz) and (not b_findwv):
        zs = config['z']
        fovs = config['fov']
        irs = config['ir'] 
        wvs = config['channel']
        return zs,fovs,irs,wvs

    # Otherwise, get all the images
    files = Path(config['raw_data_path']).rglob('*.TIFF')
    files = list(map(str,files))
    
    # Find all channel, ir, fov, and z from the file names
    filter = r"(\d{3})(?=nm).*(\B\d{2})\D(\B\d{3})\D(\B\d{2}\b)"
    param_filter = re.compile(filter)
    results = list(map(param_filter.search,files))

    # Get the list of parameters if they are needed
    #   Group 0 is the full match, so each capture group is 1 indexed
    if b_findz:
        zs = list(map(lambda x: x.group(4),results))
    else:
        zs = config['z']

    if b_findir:
        irs = list(map(lambda x: x.group(2),results))
    else:
        irs = config['ir'] 

    if b_findfov:
        fovs = list(map(lambda x: x.group(3),results))
    else:
        fovs = config['fov']

    if b_findwv:
        wvs = list(map(lambda x: x.group(1),results))
    else:
        wvs = config['channel']
    
    return sorted(list(set(zs)),key=int),sorted(list(set(fovs)),key=int),sorted(list(set(irs)),key=int),sorted(list(set(wvs)),key=int)

zs,fovs,irs,wvs = check_params()


default_message= "rule {rule}, {wildcards}, threads: {threads}"

rule all:
    threads:16
    input:
        os.path.join(config['results_path'],'brightness_report.t'),
        os.path.join(config['results_path'],'focus_report.t')


rule create_image_stack:
    threads:16
    message: default_message
    output:
        out_file = os.path.join(config['results_path'],'imgstack_{fov}.npy'),
        coord_file = os.path.join(config['results_path'],'coord_{fov}.json')
    run:
        reports.create_image_stack(os.path.join(config['raw_data_path'],config['raw_image_format']),
                                    wvs,wildcards.fov,irs,zs,output.out_file,output.coord_file)

rule brightness_report:
    threads:16
    message: default_message
    input:
        img_stack = os.path.join(config['results_path'],'imgstack_{fov}.npy'),
        coord_file = os.path.join(config['results_path'],'coord_{fov}.json')
    output:
        out=os.path.join(config['results_path'],'brightness_report_{fov}.pdf')
    run:
        reports.generate_brightness_reports(input.img_stack,input.coord_file,output.out,wildcards.fov,fovs)

rule compile_brightness_report:
    threads:16
    message: default_message
    input:
        expand(os.path.join(config['results_path'],'brightness_report_{fov}.pdf'),fov=fovs)
    output:
        os.path.join(config['results_path'],'brightness_report.t')
    shell:
        "touch \"{output}\""


rule focus_report:
    threads:16
    message: default_message
    input:
        img_stack = os.path.join(config['results_path'],'imgstack_{fov}.npy'),
        coord_file = os.path.join(config['results_path'],'coord_{fov}.json')
    output:
        out=os.path.join(config['results_path'],'focus_report_{fov}.pdf')
    run:
        reports.generate_focus_reports(input.img_stack,input.coord_file,output.out,wildcards.fov,fovs)

rule compile_focus_report:
    threads:16
    message: default_message
    input:
        expand(os.path.join(config['results_path'],'focus_report_{fov}.pdf'),fov=fovs)
    output:
        os.path.join(config['results_path'],'focus_report.t')
    shell:
        "touch \"{output}\""