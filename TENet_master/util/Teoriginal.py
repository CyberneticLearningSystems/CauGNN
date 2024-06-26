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
from numba import njit
from numba import jit
# from progressbar import ProgressBar

#* with njit first compilation takes 13 seconds, then execution is less than 1 second
#* without njit, each execution takes about 1.5 seconds
# @njit(fastmath=True, nopython=True, parallel=True) #! Not working with code below as there are no njit compatible functions, use @jit instead
@jit(fastmath=True, parallel=True)
def TE(x: np.ndarray, y: np.ndarray, pieces: int, j: int, temp1: np.ndarray):
    '''
    x, y = data for variables x and y as numpy arrays with shape (int(0.8*n), 1), with n being the number of observations in the whole timeseries
    pieces = number of bins for discretization and creation of the distribution over the observation of each feature
    j = time horizon which is used to take into account for Transfer Entropy. j is here the number of rows taken into account --> int(0.8*n) - 1
    temp1 = array with random permutation of range(j) --> np.random.shuffle(np.array(range(j)))
    '''    
    select=temp1[:j]
    d_x=np.zeros((j,4))
    d_x[:,0] = x[select+1] #Variable x(t+1)
    d_x[:,1] = x[select] #Randomly permuted data of the variable x --> x(t)
    d_x[:,2] = y[select+1] #Variable y(t+1)
    d_x[:,3] = y[select] #Randomly permuted data of the variable y --> y(t)
    
    x_max = np.max(x); x_min = np.min(x)
    y_max = np.max(y); y_min = np.min(y)
    
    delta1 = (x_max-x_min)/(2*pieces)
    delta2 = (y_max-y_min)/(2*pieces)
    
    L1 = np.linspace(x_min+delta1,x_max-delta1,pieces)
    L2 = np.linspace(y_min+delta2,y_max-delta2,pieces)
    
    #Counts the marginal probability of the two variables X and Y
    dist1=np.zeros((pieces,2))
    count=-1
    for q1 in range(pieces):
        k1=L1[q1]; k2=L2[q1]
        count+=1
        count1=0
        count2=0
        for i in range(j):
            if d_x[i,1]>=(k1-delta1) and d_x[i,1]<=(k1+delta1): #P(X(t))
                count1+=1
            if d_x[i,3]>=(k2-delta2) and d_x[i,3]<=(k2+delta2): #P(Y(t))
                count2+=1
        dist1[count,0]=count1; dist1[count,1]=count2
        
    dist1[:,0]=dist1[:,0]/np.sum(dist1[:,0]); 
    dist1[:,1]=dist1[:,1]/np.sum(dist1[:,1])
    
    #Calculates the distribution of the joint probability of the two variables X and Y, i.e. P(X given Y)
    dist2=np.zeros((pieces,pieces,3)) 
    for q1 in range(pieces):
        for q2 in range(pieces):
            k1=L1[q1]; k2=L1[q2]
            k3=L2[q1]; k4=L2[q2]
            count1=0;count2=0;count3=0
            for i1 in range(j):
                    if d_x[i1,0]>=(k1-delta1) and d_x[i1,0]<=(k1+delta1) and d_x[i1,1]>=(k2-delta1) and d_x[i1,1]<=(k2+delta1): #P(X(t+1)|X(t))            
                        count1=count1+1
                
                    if d_x[i1,2]>=(k3-delta2) and d_x[i1,2]<=(k3+delta2) and d_x[i1,3]>=(k4-delta2) and d_x[i1,3]<=(k4+delta2): #P(Y(t+1)|Y(t))
                        count2=count2+1
                    
                    if d_x[i1,1]>=(k1-delta1) and d_x[i1,1]<=(k1+delta1) and d_x[i1,3]>=(k4-delta2) and d_x[i1,3]<=(k4+delta2): #P(X(t)|Y(t))
                        count3=count3+1
                           
            dist2[q1,q2,0]=count1 
            dist2[q1,q2,1]=count2
            dist2[q1,q2,2]=count3
            
    dist2[:,:,0]=dist2[:,:,0]/np.sum(dist2[:,:,0]) #P(X(t+1)|X(t))
    dist2[:,:,1]=dist2[:,:,1]/np.sum(dist2[:,:,1]) #P(Y(t+1)|Y(t))
    dist2[:,:,2]=dist2[:,:,2]/np.sum(dist2[:,:,2]) #P(X(t)|Y(t)) 
    
    dist3=np.zeros((pieces,pieces,pieces,2))
    

    for q1 in range(pieces):
        for q2 in range(pieces):
            # TODO: move this q3 loop into 
            for q3 in range(pieces):
                k1=L1[q1]; k2=L1[q2]; k3=L1[q3]
                k4=L2[q1]; k5=L2[q2]; k6=L2[q3]
                count1=0; count2=0
                for i1 in range(j):
                    if d_x[i1,0]>=(k1-delta1) and d_x[i1,0]<=(k1+delta1) and d_x[i1,1]>=(k2-delta1) and d_x[i1,1]<=(k2+delta1) and d_x[i1,3]>=(k6-delta2) and d_x[i1,3]<=(k6+delta2):  #P(X(t+1)|X(t),Y(t))
                        count1=count1+1
                   
                    if d_x[i1,2]>=(k4-delta2) and d_x[i1,2]<=(k4+delta2) and d_x[i1,3]>=(k5-delta2) and d_x[i1,3]<=(k5+delta2) and d_x[i1,1]>=(k3-delta1) and d_x[i1,1]<=(k3+delta1):  #P(Y(t+1)|Y(t),X(t))
                        count2=count2+1

                dist3[q1,q2,q3,0]=count1; dist3[q1,q2,q3,1]=count2
    
    #! this throws an RuntimeWarning: invalid value encountered in true_divide
    dist3[:,:,:,0]=dist3[:,:,:,0]/np.sum(dist3[:,:,:,0]) #P(X(t+1)|X(t),Y(t))
    dist3[:,:,:,1]=dist3[:,:,:,1]/np.sum(dist3[:,:,:,1]) #P(Y(t+1)|Y(t),X(t))
    
    sum_f_1=0
    sum_f_2=0
    for k1 in range(pieces):
        for k2 in range(pieces):
            if dist2[k1,k2,1]!=0 and dist1[k2,1]!=0:
                sum_f_1 = sum_f_1-dist2[k1,k2,1] * np.log2(dist2[k1,k2,1]/dist1[k2,1]) #P(Y(t+1),Y(t)) * log2(P(Y(t+1),Y(t))/P(Y(t)))

            if dist2[k1,k2,0]!=0 and dist1[k2,0]!=0:
                sum_f_2 = sum_f_2-dist2[k1,k2,0] * np.log2(dist2[k1,k2,0]/dist1[k2,0]) #P(X(t+1),X(t)) * log2(P(X(t+1),X(t))/P(X(t)))
    sum_s_1=0
    sum_s_2=0
    for k1 in range(pieces):
        for k2 in range(pieces):
            # TODO: move k3 for-loop into the above loop
            for k3 in range(pieces):
                if dist3[k1,k2,k3,1]!=0 and dist2[k3,k2,2]!=0:
                    sum_s_1 = sum_s_1-dist3[k1,k2,k3,1] * np.log2(dist3[k1,k2,k3,1]/dist2[k3,k2,2]) #P(Y(t+1)|Y(t),X(t)) * log2(P(Y(t+1)|Y(t),X(t))/P(X(t)|Y(t)))

                if dist3[k1,k2,k3,0]!=0 and dist2[k2,k3,2]!=0:
                    sum_s_2 = sum_s_2-dist3[k1,k2,k3,0] * np.log2(dist3[k1,k2,k3,0]/dist2[k2,k3,2]) #P(X(t+1)|X(t),Y(t)) * log2(P(X(t+1)|X(t),Y(t))/P(X(t)|Y(t)))
    
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
        try:
            data = pd.read_csv(datapath, delimiter=',')
            data.drop(columns=['YEAR', 'QUARTER', 'AIRLINE_ID', 'UNIQUE_CARRIER_NAME'], inplace=True)
        except KeyError:
            data = pd.read_csv(datapath, delimiter=',')
            data.drop(columns=['YEAR', 'QUARTER'], inplace=True)
    except KeyError:
        data = pd.read_csv(datapath, delimiter=';')
        data.drop(columns=['YEAR', 'QUARTER', 'AIRLINE_ID', 'UNIQUE_CARRIER_NAME'], inplace=True)
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
    n = np.sum(range(data.shape[1]))
    t = 0
    # bar = ProgressBar('Processing', maxval=703, suffix='%(percent)d%%')
    # bar = Bar('Processing', max=703, fill='@', suffix='%(percent)d%%')
    start_timer()
    for var1 in range(data.shape[1]):
        for var2 in range(var1+1, data.shape[1]):
            print(f'     {np.round(((t/n)/2)*100, 2)}%        ')
            time.sleep(0.0000001)
            # bar.next()
            # passes 80% of the data to the TE function for var1 and var2
            temp1 = np.array(range(L-1))
            np.random.shuffle(temp1)
            te1, te2 = TE(x = data[:L,var1], y = data[:L,var2], pieces = 50, j = L-1, temp1=temp1)
            t += 1
            if te1 >= te2:
                A[var1,var2] = te1-te2
            if te1 < te2:
                A[var2,var1] = te2-te1
            t += 1
    end_timer()
    # bar.finish()
    return np.array(A, dtype=np.float32)


def _save_matrix(A, outputpath):
    file = open(outputpath, 'w')
    for i in range(A.shape[1]):
        for j in range(A.shape[1]):
            file.write(str(A[i,j])+' ')
        file.write('\n')
    file.close()


def calculate_te_matrix(datapath, outputfolder):
    outputpath = os.path.join(outputfolder, f'{os.path.splitext(os.path.basename(datapath))[0]}_TE.txt')
    data = _dataloader(datapath)
    A = _te_calculation(data)
    _save_matrix(A, outputpath)
    print(f'Transfer Entropy matrix saved to {outputpath}')
    return A


def start_timer():
    global _start_time
    _start_time = time.time()


def end_timer():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print(f'Time: {t_hour}:{t_min}:{t_sec}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='data/exchange_rate.txt')
    parser.add_argument('--outputfolder', type=str, default='TE')
    args = parser.parse_args()

    _ = calculate_te_matrix(args.datapath, args.outputfolder)
