#画bimodal的真实数据与拟合数据图
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
SIZE=200
rate=0.45
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


files_para = './data/ForNorData_MSTd/'
files_r = './data/AllData_MSTd/AllData/'
both_azimuth_files = os.listdir(files_para)
r_files = os.listdir(files_r)

VEST = list(range(0, 360 + 45, 45))  # H: vest azimuth
VIS = list(range(0, 360 + 45, 45))  # H: vis azimuth

global vest_a, vis_a, vest_e, vis_e, n1, n2, n3, e, c_vest, c_vis, n0, \
    S_vest_m, S_vis_m,RawR_trial,RawR
RawR = []  # firing curve, mean of neuron responses under different stimulus conditions, 就是我们最后要fit的y值（平均数）
RawR_trial = []  # neuron responses under different stimulus conditions, 就是我们最后要fit的y值（实际trial的y值）
for f in both_azimuth_files:


    temp_file = format_matfile(files_r + 'SmoothData_' + f[:-15] + '.mat', 'CueConflictDataSmooth')
    RawR_trial.append(temp_file['resp_trial_conflict'])
    RawR.append(temp_file['resp_conflict'])
SIZE=len(RawR_trial)
corr_list = [(np.triu(np.corrcoef([i.ravel() for i in RawR_trial[n]])).sum() - RawR_trial[n].shape[0]) / comb(
        RawR_trial[n].shape[0], 2) for n in range(SIZE)]
selected_neurons = [i for i in range(SIZE) if
                        corr_list[i] > rate]  # select good neurons: correlation between trials over 0.4
print(selected_neurons)
print(len(selected_neurons))
RawR = np.array(RawR)[selected_neurons]
RawR_trial = np.array(RawR_trial)[selected_neurons]
# selected_neurons=list(range(200))

RawR_fit=[]
for i in range(len(selected_neurons)):
    path='./result/all/'+str(i)+'.xls'
    u=read(path)
    RawR_fit.append(u)
# for i in range(len(selected_neurons)):
#     ax = plt.scatter(np.ravel(RawR_fit[i]), np.ravel(RawR[i]), s=2, marker='*')
#     plt.xlabel('Fitted R')
#     plt.ylabel('Actual R')
#     plt.plot(list(range(120)), list(range(120)), 'r--', linewidth=2)
#     ax.figure.set_size_inches(8, 8)
#     high = max(np.ravel(RawR[i]).max(), np.ravel(RawR_fit[i]).max())
#     low = min(np.ravel(RawR[i]).min(), np.ravel(RawR_fit[i]).min())
#     plt.ylim((low, high))
#     plt.xlim((low, high))
#     path='C:/Users/76774/Desktop/test/image_5_nobound/'+str(i)+'.jpg';
#     plt.savefig(path,dopi=600);
#     plt.show()
ax = plt.scatter(np.ravel(RawR_fit), np.ravel(RawR), s=2, marker='*')
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 20,
}
plt.tick_params(labelsize=20)
plt.xlabel('Fitted R',font2)
plt.ylabel('Actual R',font2)
plt.plot(list(range(140)), list(range(140)), 'r--', linewidth=2)
# plt.legend(['Diagnal fitted line', 'Data'])
ax.figure.set_size_inches(8, 8)
high = max(np.ravel(RawR).max(), np.ravel(RawR_fit).max())
low = min(np.ravel(RawR).min(), np.ravel(RawR_fit).min())
plt.ylim((low, high))
plt.xlim((low, high))
plt.title('bimoda',font2)
path='./result/image/sum.jpg'
plt.savefig(path,dpi=600)
plt.show()