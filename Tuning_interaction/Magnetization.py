
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from fastpip import pip
from scipy.signal import savgol_filter    
import scipy

# This python file will contain the main functions needed in order to perform the analysis of the videos after 
# tracking. With all this functions I will be able to find the bifurcation point, eather a particles has choose the upper
# or lower channel, the magnetization and so on. Those functions work correctly if the tracking is performed correctly,
# particles must be tracked during the whole main channel. I generated a function that raise the alert that the analysis 
# could be wrong. 

def y_bifurcation_position(df):
    """In this function I will locate the bifurcation position"""
    
    y, x , _ = plt.hist(df['y'], bins = 40, )
    y_bif = x[np.argmax(y)]
    
    return y_bif, df

def x_bifurcation_position(df, y_bif):
    """In this function I will find the x value of the bifurcation"""
    
    particles = df.particle.unique()
    points = []
    for i in particles:
        
        particle = df.query('particle =='+str(i))

        data = np.stack((particle.x, particle.y), axis=-1)
        points.append(pip(data,4))
    
    points_ar = np.array(points)
    new_df = points_ar.reshape((len(points_ar[0])*len(points_ar),2))
    df = pd.DataFrame(data=new_df, columns = ['x', 'y'])
    y_bifurcation_plus = y_bif+10
    y_bifurcation_minus = y_bif-10

    df_filtered = df.query('y < '+str(y_bifurcation_plus)+' & y > '+str(y_bifurcation_minus))
    x_bif = df_filtered.x.min()
    return x_bif

def up_or_down(df, y_bif):
    
    """This function indicates which channel the particle has choose.
    The information is stored in the column bif."""
    
    df_trj = df[df.duplicated(subset=['frame'], keep=False)] 

    df_trj['bif'] = np.NaN
    for p,df in df_trj.groupby('particle'):
        
        if df_trj.iloc[-1].y > y_bif:
            df_trj.loc[df_trj.particle==p,'bif'] = 'UP'
        else:
            df_trj.loc[df_trj.particle==p,'bif']= 'DOWN'
        
    return df_trj

def create_pairs(df, frame):
    """This function generates pair of particles. A pair of particle is defined 
    when one particles enter to the channel after the other. So, consecutive
    particles will be part of a pair.
    This function works for one frame"""

    df_f = df.query('frame == '+str(frame)) # First select a frame

    df_order = df_f.sort_values(by = ['x'])       # Order the x column in ascendent values, 
                                          # this will be the order of particles entering the chanel. 
    new_df = []
    keys = []
    p1 = []
    p2 = []
    j = 2
    
    for i in range(0, len(df_order)-1):       
        
        p1.append(df_order.iloc[i:j].particle.values[0])
        p1.append(df_order.iloc[i:j].particle.values[0])


        p2.append(df_order.iloc[i:j].particle.values[1])
        p2.append(df_order.iloc[i:j].particle.values[1])

        new_df.append(df_order.iloc[i:j])  # Join consecutive rows, generating pairs
      

        keys.append(i)
        j = j +1
    p1 = pd.DataFrame({'p1':p1})
    p2 = pd.DataFrame({'p2':p2})

    final_df = pd.concat(new_df, keys = keys, names = ['pairs','index'])
    pairs = pd.concat([p1,p2], axis = 1)
    
    final_df['p1'] = pairs['p1'].values
    final_df['p2'] = pairs['p2'].values
    
    return final_df

def magnetization(df, frame):
    df = df.query('frame == '+str(frame))
    magnetization = []
    for ind,group in df.groupby(['pairs']):

        if len(np.unique(group.bif)) == 2:

            magnetization.append(0)
            magnetization.append(0)

        else:

            magnetization.append(1)
            magnetization.append(1)


    df['magnetization'] = magnetization
    return df

def d(df , pix = float, um = float):  
    "With this function I will compute the distance among particles."
    
    df['distance'] = np.NaN # Crate new column
    d = []
    grouped = df.groupby(['frame','pairs'])
    for name,group in grouped:
        
        d.append((np.sqrt((group.x.diff(periods=-1).values)**2 + (group.y.diff(periods=-1).values)**2)[0])*(um/pix))
        d.append((np.sqrt((group.x.diff(periods=-1).values)**2 + (group.y.diff(periods=-1).values)**2)[0])*(um/pix))
    
    df['distance'] = d
    
    return df

def check_analysis(trj_initial,last_df_analyzed):
    
    n_of_particles = len(trj_initial['particle'].unique())
    n_of_pairs = len(last_df_analyzed.groupby(['p1','p2']).size())
    
    if (n_of_particles == n_of_pairs+1):
        
        print("Analysis succesfully performed of video")
    else:
        print("Something was wrong during tracking or analysis. We detect something different to N particles and N-1 pairs")
        
def plot(trj,x_bif,y_bif):
    
    fig, ax1 = plt.subplots(1,1,figsize=(15,16))

    for p,trj_p in trj.groupby("particle"):
        ax1.plot(trj_p.x, trj_p.y, 'o', alpha = 0.1, markersize = 20)
        plt.axhline(y=y_bif, color='r', linestyle='-')
        plt.axvline(x=x_bif, color='r', linestyle='-')
    