from re import L
from turtle import shape
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np
#load csv files
#returns files names as well to preserve laser channel information
def load_data(path:Path) -> tuple:
    files=[os.path.join(path,file) for file in os.listdir(path)]
    dfs=[]
    for f in files:
        if ".csv" in f.lower(): 
            df=pd.read_csv(f)
            dfs.append(df)
    return tuple(dfs)


def find_rect(fovs:int):
    area=fovs
    w=area
    l=2
    while l <=w :    
        if (area/l)%1==0: w=area/l     
        l+=1
    return int(w)

def min_max_scale_percent(df,val):
    max_spot=np.max(df["total_spots"])
    min_spot=np.min(df["total_spots"])
    return round((val-min_spot)/(max_spot-min_spot)*100,2) 

#shows simple heatmap with median spot count per tile
#counts are scaled between 1-100 just to get a sense of a good tile
def plot_heatmap(df):
    total_fov=np.max(df["FOV"])
    
    w=find_rect(total_fov)
    l=int(total_fov/w)
    heatmap=np.empty(shape=(w,l),dtype=int)
    index=1
    for y in range(heatmap.shape[0]):
        for x in range(heatmap.shape[1]):
            print(index)
            val=min_max_scale_percent( 
                df,
                np.median(df["total_spots"][df["FOV"] == index])
            )
            heatmap[y][x]=val
            index+=1
    sns.heatmap(data=heatmap,annot=True)
    plt.show()



#plots 2*num_tiles bar graphs each showing spots captured in a round for that tile  
def plot_grid(ch1,ch2,savePath:Path):
    os.chdir(savePath)
    total_fov=np.max(ch1["FOV"])
    ymax=np.max(ch1["total_spots"]) if np.max(ch1["total_spots"])>np.max(ch2["total_spots"])  else np.max(ch2["total_spots"])
    fig, axs= plt.subplots(figsize=(16,120),nrows=total_fov,ncols=2,tight_layout=True)
    
    for i in range(axs.shape[0]):
        axs[i][0].set_title(f"FOV: {i+1}")
        axs[i][0].set_ylabel(f"# of spots", fontsize=15)
        axs[i][0].set_xlabel("imaging round", fontsize=10)
        axs[i][0].set_ylim(ymax)
        axs[i][0].bar(
            ch1["imaging_round"][ch1["FOV"]==i+1],
            ch1["total_spots"][ch1["FOV"]==i+1]
        )
        axs[i][0].invert_yaxis()
    axs[0][0].set_title("647\nFOV: 1")
    for i in range(axs.shape[0]):
        axs[i][1].set_title(f"FOV: {i+1}")
        axs[i][1].set_ylabel(f"# of spots", fontsize=15)
        axs[i][1].set_xlabel("imaging round", fontsize=10)
        axs[i][1].set_ylim(ymax)
        axs[i][1].bar(
            ch2["imaging_round"][ch2["FOV"]==i+1],
            ch2["total_spots"][ch2["FOV"]==i+1]
        )
        axs[i][1].invert_yaxis()
    axs[0][1].set_title("750\nFOV: 1")
    
    plt.tight_layout(pad=0.4,w_pad=0.5,h_pad=1)
    plt.savefig(os.path.join(savePath, "spots_accross_FOVs.pdf"),format="pdf")
   
    


if __name__ == "__main__":
    path= r"C:\Users\Isaac von Riedemann\Downloads\counting_spots_across_FOVs_dataset_XP2059_647"
    dfs=load_data(path)
    
    plot_grid(dfs[0],dfs[1],path)