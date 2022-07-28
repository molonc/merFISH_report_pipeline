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
        if ".csv" in f.lower(): df=pd.read_csv(f)
        dfs.append((f,df))
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

    
def plot_grid(df):
    total_fov=np.max(df["FOV"])
    print(find_rect(42))
    fig, axs= plt.subplots(figsize=(3,2),nrows=7,ncols=6)
    # g=sns.catplot(
    #     data=df,
    #     x="imaging_round",
    #     y="total_spots",
    #     kind="bar",
    #     row="FOV",
    #     height=2.5,
    #     aspect=.5,
    #     margin_titles=True
    # )
    

    plt.show()


if __name__ == "__main__":
    path= r"C:\Users\Isaac von Riedemann\Downloads\counting_spots_across_FOVs_dataset_XP2059_647"
    dfs=load_data(path)
    df=dfs[0][1]
    print(df["total_spots"][df["FOV"]==14])
    plot_heatmap(dfs[0][1])