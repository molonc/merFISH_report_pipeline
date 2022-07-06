import os
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import skimage
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type = str, help = 'Input directory to csv files')
    parser.add_argument('--output_dir', type = str, help = 'Output directory to store the results')
    parser.add_argument('--exp_name', type = str, help = 'Experiment name, e.g.: XP2059')
    parser.add_argument('--ch', type = int, help='Channel of concern')
    
    args = parser.parse_args()

    if not args.output_dir:
        args.output_dir = args.input_dir

    f_list = irlist(args.input_dir, '*.csv')

    num_zslices = len(pd.read_csv(f_list[0]))

    num_irs = len(f_list)

    fov = f_list[0].split('_')[-3][3:]

    spots, xlist, X = get_xy(f_list, num_zslices)

    ymax = np.amax(spots)
    
    fig, axes = plt.subplots(nrows=num_irs, ncols=1, figsize=(30, 60))
    fig.suptitle(f'Counting spots for {args.exp_name}, FOV = {fov}, channel = {args.ch} nm', fontsize = 40)

    
    for i, ax in enumerate(axes.flat): 
        ax.set_xticks(X, xlist[i], fontsize = 25)
        ax.set_xticklabels(xlist[i], rotation = 45)
        ax.set_ylabel(f'# spots in IR {i+1}', fontsize = 28, fontweight="bold")
        ax.set_ylim([0,ymax])
        ax.tick_params(axis='y', labelsize=25)
        ax.bar(X, spots[:,i], width = args.bar_width)
        
   
    plt.tight_layout()
    
    fig.savefig(args.output_dir+f'/{args.exp_name}_{fov}_{args.ch}.pdf', transparent = 'False', format = 'pdf')

def irlist(idir, match_str):
    idir = Path(idir)
    return [str(i) for i in sorted(list(idir.rglob(match_str)))]

def get_xy(csv_list, zslices):
    num_ir = len(csv_list)
    spots_arr = np.zeros(shape=(zslices,num_ir))
    x_ticks = []
    for i in range(num_ir):
        df = pd.read_csv(csv_list[i])
        spots_arr[:,i] = df['Spots'].to_numpy()
        x = [s.replace('merFISH_', "") for s in sorted(list(df['Image']))]
        x_ticks.append(x) 
    X = np.arange(zslices)
    return spots_arr, x_ticks, X

if __name__ == '__main__':
    main()
    #python counting_spots_z_display.py --input_dir /Users/ythapliyal/Counting_spots_old/XP2059/647 --exp_name XP2059 --ch 647

