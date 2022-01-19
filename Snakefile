from pathlib import Path
import re
import os
import reports


configfile: "config.json"


def download_azure():
    
    if config["isRemote"]=="N" or os.path.exists(os.path.join(config['results_path'],'data')):
        return None
    
    shell(config["azure_command"].format(config["raw_data_path"],os.path.join(config['results_path'],'data')))
    
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


download_azure()
zs,fovs,irs,wvs,full_raw_path = check_params()

print(wvs)

default_message= "rule {rule}, {wildcards}, threads: {threads}"

rule all_done:
    input:
        os.path.join(config['results_path'],'brightness_report.t'),
        os.path.join(config['results_path'],'focus_report.t'),
    output:
        os.path.join(config['results_path'],'all_done.t')
    run:
        if config["isRemote"]=="Y":
            shell("rm -rf \"{}\"".format(os.path.join(config['results_path'],'data')))
        if config["delete_stack"]=="Y":
            shell("rm -f \"{}\"".format(os.path.join(config['results_path'],'img*')))
        shell("touch \"{output[0]}\"")

rule all:
    threads:1
    input:
        os.path.join(config['results_path'],'all_done.t')


def isRemote(wildcards):
    
    if config["isRemote"]=="N":
        return config["raw_data_path"]
    else:
        return directory(os.path.join(config['results_path'],'data'))

rule create_image_stack:
    threads:16
    message: default_message
    input:
        isRemote
    output:
        out_file = os.path.join(config['results_path'],'imgstack_{fov}.npy'),
        coord_file = os.path.join(config['results_path'],'coord_{fov}.json')
    run:
        reports.create_image_stack(os.path.join(full_raw_path,config['raw_image_format']),
                                    wvs,wildcards.fov,irs,zs,output.out_file,output.coord_file)

rule brightness_report:
    threads:1
    message: default_message
    input:
        img_stack = os.path.join(config['results_path'],'imgstack_{fov}.npy'),
        coord_file = os.path.join(config['results_path'],'coord_{fov}.json')
    output:
        out=os.path.join(config['results_path'],'brightness_report_{fov}.pdf')
    run:
        reports.generate_brightness_reports(input.img_stack,input.coord_file,output.out,wildcards.fov,fovs)

rule compile_brightness_report:
    threads:1
    message: default_message
    input:
        expand(os.path.join(config['results_path'],'brightness_report_{fov}.pdf'),fov=fovs)
    output:
        os.path.join(config['results_path'],'brightness_report.t')
    shell:
        "touch \"{output}\""


rule focus_report:
    threads:1
    message: default_message
    input:
        img_stack = os.path.join(config['results_path'],'imgstack_{fov}.npy'),
        coord_file = os.path.join(config['results_path'],'coord_{fov}.json')
    output:
        out=os.path.join(config['results_path'],'focus_report_{fov}.pdf'),
        out_csvs = os.path.join(config['results_path'],'focus_report_{fov}.csv')
    run:
        reports.generate_focus_reports(input.img_stack,input.coord_file,output.out,output.out_csvs,wildcards.fov,fovs)

rule compile_focus_report:
    threads:1
    message: default_message
    input:
        files = expand(os.path.join(config['results_path'],'focus_report_{fov}.pdf'),fov=fovs),
        csvs = expand(os.path.join(config['results_path'],'focus_report_{fov}.csv'),fov=fovs)
    output:
        combined = os.path.join(config['results_path'],'focus_report_all_fov.pdf'),
        out = os.path.join(config['results_path'],'focus_report.t')
    run:
        reports.compile_focus_report(input.csvs,output.combined,irs,wvs)
        shell("touch \"{output.out}\"")