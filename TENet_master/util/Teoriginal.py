# -*- coding: utf-8 -*-
import os
import time
import argparse
import os
import time
import argparse
import numpy as np
import pandas as pd
import pandas as pd
from numba import jit
# from progressbar import ProgressBar

#! this was just commented out because it was causing errors with np.random.shuffle not having an implementation for temp1 
# @jit(nopython=True)
def TE(x: np.ndarray, y: np.ndarray, pieces: int, j: int):
    '''
    x, y = data for variables x and y as numpy arrays with shape (int(0.8*n), 1), with n being the number of observations in the whole timeseries
    pieces = arbitrary integer (no clue what for)
    j = int(0.8*n) - 1 (no clue why -1, but it's used to iterate through the data)
    '''
    # J can be used to iterate through the data
    d_x=np.zeros((j,4))
    sit=len(x)
    temp1=np.array(range(sit-1))
    np.random.shuffle(temp1)
    select=np.array(temp1[:j])
    
    d_x[:,0] = x[select+1]
    d_x[:,1] = x[select]
    d_x[:,2] = y[select+1]
    d_x[:,3] = y[select]
    
    x_max = np.max(x); x_min = np.min(x)
    y_max = np.max(y); y_min = np.min(y)
    
    delta1 = (x_max-x_min)/(2*pieces)
    delta2 = (y_max-y_min)/(2*pieces)
    
    L1 = np.linspace(x_min+delta1,x_max-delta1,pieces)
    L2 = np.linspace(y_min+delta2,y_max-delta2,pieces)
    
    dist1=np.zeros((pieces,2))
    count=-1
    
    for q1 in range(pieces):
        k1=L1[q1]; k2=L2[q1]
        count+=1
        count1=0
        count2=0
        for i in range(j):
            if d_x[i,1]>=(k1-delta1) and d_x[i,1]<=(k1+delta1):
                count1+=1
            if d_x[i,3]>=(k2-delta2) and d_x[i,3]<=(k2+delta2):
                count2+=1
        dist1[count,0]=count1; dist1[count,1]=count2
        
    dist1[:,0]=dist1[:,0]/sum(dist1[:,0]); dist1[:,1]=dist1[:,1]/sum(dist1[:,1])
    
    dist2=np.zeros((pieces,pieces,3))
    for q1 in range(pieces):
        for q2 in range(pieces):
            k1=L1[q1]; k2=L1[q2]
            k3=L2[q1]; k4=L2[q2]
            count1=0;count2=0;count3=0
            for i1 in range(j):
                    if d_x[i1,0]>=(k1-delta1) and d_x[i1,0]<=(k1+delta1) and d_x[i1,1]>=(k2-delta1) and d_x[i1,1]<=(k2+delta1):
                        count1=count1+1
                
                    if d_x[i1,2]>=(k3-delta2) and d_x[i1,2]<=(k3+delta2) and d_x[i1,3]>=(k4-delta2) and d_x[i1,3]<=(k4+delta2):
                        count2=count2+1
                    
                    if d_x[i1,1]>=(k1-delta1) and d_x[i1,1]<=(k1+delta1) and d_x[i1,3]>=(k4-delta2) and d_x[i1,3]<=(k4+delta2):
                        count3=count3+1
                           
            dist2[q1,q2,0]=count1 
            dist2[q1,q2,1]=count2
            dist2[q1,q2,2]=count3
            
    dist2[:,:,0]=dist2[:,:,0]/np.sum(dist2[:,:,0])
    dist2[:,:,1]=dist2[:,:,1]/np.sum(dist2[:,:,1])
    dist2[:,:,2]=dist2[:,:,2]/np.sum(dist2[:,:,2])
    
    dist3=np.zeros((pieces,pieces,pieces,2))
    
    for q1 in range(pieces):
        for q2 in range(pieces):
            for q3 in range(pieces):
                k1=L1[q1]; k2=L1[q2]; k3=L1[q3]
                k4=L2[q1]; k5=L2[q2]; k6=L2[q3]
                count1=0; count2=0
                for i1 in range(j):
                    if d_x[i1,0]>=(k1-delta1) and d_x[i1,0]<=(k1+delta1) and d_x[i1,1]>=(k2-delta1) and d_x[i1,1]<=(k2+delta1) and d_x[i1,3]>=(k6-delta2) and d_x[i1,3]<=(k6+delta2):  
                        count1=count1+1
                   
                    if d_x[i1,2]>=(k4-delta2) and d_x[i1,2]<=(k4+delta2) and d_x[i1,3]>=(k5-delta2) and d_x[i1,3]<=(k5+delta2) and d_x[i1,1]>=(k3-delta1) and d_x[i1,1]<=(k3+delta1): 
                        count2=count2+1

                dist3[q1,q2,q3,0]=count1; dist3[q1,q2,q3,1]=count2
    
    #! this throws an RuntimeWarning: invalid value encountered in true_divide
    dist3[:,:,:,0]=dist3[:,:,:,0]/np.sum(dist3[:,:,:,0])
    dist3[:,:,:,1]=dist3[:,:,:,1]/np.sum(dist3[:,:,:,1])
    
    sum_f_1=0
    sum_f_2=0
    for k1 in range(pieces):
        for k2 in range(pieces):
            if dist2[k1,k2,1]!=0 and dist1[k2,1]!=0:
                sum_f_1 = sum_f_1-dist2[k1,k2,1] * np.log2(dist2[k1,k2,1]/dist1[k2,1])

            if dist2[k1,k2,0]!=0 and dist1[k2,0]!=0:
                sum_f_2 = sum_f_2-dist2[k1,k2,0] * np.log2(dist2[k1,k2,0]/dist1[k2,0])
    
    sum_s_1=0
    sum_s_2=0
    for k1 in range(pieces):
        for k2 in range(pieces):
            for k3 in range(pieces):
                if dist3[k1,k2,k3,1]!=0 and dist2[k3,k2,2]!=0:
                    sum_s_1 = sum_s_1-dist3[k1,k2,k3,1] * np.log2(dist3[k1,k2,k3,1]/dist2[k3,k2,2])

                if dist3[k1,k2,k3,0]!=0 and dist2[k2,k3,2]!=0:
                    sum_s_2 = sum_s_2-dist3[k1,k2,k3,0] * np.log2(dist3[k1,k2,k3,0]/dist2[k2,k3,2])
    
    en_1_2=sum_f_1-sum_s_1
    en_2_1=sum_f_2-sum_s_2
    
    return en_1_2, en_2_1


def _dataloader(datapath: str):
    if datapath.endswith('.txt'):
        return _dataloader_standard(datapath)
    elif datapath.endswith('.csv'):
        return _dataloader_form41(datapath)
    else:
        raise ValueError('Invalid data file format. Must be .txt or .csv')


def _dataloader_standard(datapath):
    return np.loadtxt(args.datapath, delimiter=',')


def _dataloader_form41(datapath):
    try:
        data = pd.read_csv(datapath, delimiter=',')
        _ = data.pop('AIRLINE_ID')
        _ = data.pop('YEAR')
        _ = data.pop('MONTH')
    except KeyError:
        data = pd.read_csv(datapath, delimiter=';')
        _ = data.pop('AIRLINE_ID')
        _ = data.pop('YEAR')
        _ = data.pop('MONTH')
    data = np.array(data, dtype=float)
    return data


def _te_calculation(data):
    '''
    Calculate the Transfer Entropy matrix for a given dataset.

    Parameters:
    data (np.ndarray): The dataset to calculate the Transfer Entropy matrix for. Is two-dimensional, with rows representing samples 
    and columns representing variables.

    Returns:
    np.ndarray: The Transfer Entropy matrix for the given dataset. Is two-dimensional (n x n) matrix, with rows and columns representing variables.
    '''
    #! progress bar was commented out, not fixed.
    A = np.eye(data.shape[1])
    L = 0.8*data.shape[0]
    L = int(L)
    t = 0
    # bar = ProgressBar('Processing', maxval=703, suffix='%(percent)d%%')
    # bar = Bar('Processing', max=703, fill='@', suffix='%(percent)d%%')
    for var1 in range(data.shape[1]):
        for var2 in range(var1+1, data.shape[1]):
            t += 1
            print('     ',t/7.03,'%\r')
            time.sleep(0.0000001)
            # bar.next()
            # passes 80% of the data to the TE function for var1 and var2
            te1, te2 = TE(x = data[:L,var1], y = data[:L,var2], pieces = 50, j = L-1)
            if te1 >= te2:
                A[var1,var2] = te1-te2
            if te1 < te2:
                A[var2,var1] = te2-te1
    # bar.finish()
    return A


def _save_matrix(A, outputpath):
    file = open(outputpath, 'a+')
    for i in range(A.shape[1]):
        for j in range(A.shape[1]):
            file.write(str(A[i,j])+' ')
        file.write('\n')
    file.close()


def calculate_te_matrix(datapath, outputfolder):
    outputpath = os.path.join(args.outputfolder, f'{os.path.splitext(os.path.basename(args.datapath))[0]}_TE.txt')
    data = _dataloader(datapath)
    A = _te_calculation(data)
    _save_matrix(A, outputpath)
    print(f'Transfer Entropy matrix saved to {outputfolder}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='data/exchange_rate.txt')
    parser.add_argument('--outputfolder', type=str, default='TE')
    args = parser.parse_args()

    calculate_te_matrix(args.datapath, args.outputfolder)
