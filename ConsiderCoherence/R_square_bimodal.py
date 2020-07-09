#计算bimodal数据的R_square
import numpy as np
import scipy.io as sio
import os
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.special import comb
import xlwt
import xlrd
import CoherenceSelect
SIZE=158
def read(path):
    book = xlrd.open_workbook(path)
    sheet1 = book.sheets()[0]
    nrows = sheet1.nrows
    ncols = sheet1.ncols
    current_vect = []
    for i in range(nrows):
        u=[]
        for j in range(ncols):
            u.append(sheet1.cell(i, j).value)
        current_vect.append(u)
    return current_vect

def format_matfile(path, name):
    a = sio.loadmat(path)[name]
    return dict(zip(a.dtype.names, [i for i in a.item()]))

files_para = '../data/ForNorData_MSTd/'
files_r = '../data/AllData_MSTd/AllData/'
both_azimuth_files = os.listdir(files_para)
r_files = os.listdir(files_r)

VEST = list(range(0, 360 + 45, 45))  # H: vest azimuth
VIS = list(range(0, 360 + 45, 45))  # H: vis azimuth

global vest_a, vis_a, vest_e, vis_e, n1, n2, n3, e, c_vest, c_vis, n0, \
    S_vest_m, S_vis_m,RawR_trial,RawR
RawR = []  # firing curve, mean of neuron responses under different stimulus conditions, 就是我们最后要fit的y值（平均数）
RawR_trial = []  # neuron responses under different stimulus conditions, 就是我们最后要fit的y值（实际trial的y值）
collection=CoherenceSelect.CoherenceSelect()
for f in both_azimuth_files:
    if f[:-18] in collection:
        temp_file = format_matfile(files_r + 'SmoothData_' + f[:-15] + '.mat', 'CueConflictDataSmooth')
        RawR_trial.append(temp_file['resp_trial_conflict'])
        RawR.append(temp_file['resp_conflict'])
SIZE=len(RawR)
corr_list = [(np.triu(np.corrcoef([i.ravel() for i in RawR_trial[n]])).sum() - RawR_trial[n].shape[0]) / comb(
    RawR_trial[n].shape[0], 2) for n in range(SIZE)]
global selected_neurons
selected_neurons = [i for i in range(SIZE) if
                    corr_list[i] > 0.4]
RawR = np.array(RawR)[selected_neurons]
RawR_trial = np.array(RawR_trial)[selected_neurons]
print(np.array(RawR).shape)
# selected_neurons=list(range(200))
RawR_fit=[]
for i in range(len(selected_neurons)):
    path='../resultConsiderCoherence/all/'+str(i)+'.xls'
    u=read(path)
    RawR_fit.append(u)



sum1=sum(sum(sum(RawR)))
ave=sum1/(len(RawR_fit)*9*9)
SSR=0
SST=0
SSE=0
for k in range(len(RawR_fit)):
    for i in range(9):
        for j in range(9):
            SSE=SSE+(RawR[k][i][j]-RawR_fit[k][i][j])*(RawR[k][i][j]-RawR_fit[k][i][j])/(SIZE*9*9)
            SSR=SSR+(RawR_fit[k][i][j]-ave)*(RawR_fit[k][i][j]-ave)/(SIZE*9*9)
            SST=SST+(RawR[k][i][j]-ave)*(RawR[k][i][j]-ave)/(SIZE*9*9)
print(SSR/SST)
print(1-SSE/SST)
y=[]
# for k in range(len(selected_neurons)):
#     sum1=sum(sum(RawR[k]))
#     ave=sum1/81
#     SSR=0
#     SST=0
#     SSE=0
#     for i in range(9):
#         for j in range(9):
#             SSE=SSE+(RawR[k][i][j]-RawR_fit[k][i][j])*(RawR[k][i][j]-RawR_fit[k][i][j])/81
#     for i in range(9):
#         for j in range(9):
#             SSR=SSR+(RawR_fit[k][i][j]-ave)*(RawR_fit[k][i][j]-ave)/(9*9)
#             SST=SST+(RawR[k][i][j]-ave)*(RawR[k][i][j]-ave)/(9*9)
#     # print('SSE/SST:',SSE/SST)
    # print('SSR/SST:',SSR/SST)
    # print(1-SSE/SST)
    # y.append((1-SSE/SST))
# x=[]
# y1=[]
# for i in range(200):
#     if y[i]>0:
#         x.append(i)
#         y1.append(y[i])
#
# plt.scatter(x,y1)
# plt.show()