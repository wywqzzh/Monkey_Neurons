#计算unimodal数据的R_square
import numpy as np
import scipy.io as sio
import os
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.special import comb
import xlrd
import CoherenceSelect
SIZE=158
files_para = '../data/ForNorData_MSTd/'
files_r = '../data/AllData_MSTd/AllData/'
both_azimuth_files = os.listdir(files_para)
r_files = os.listdir(files_r)
RawR_trial = []  # neuron responses under different stimulus conditions
RawR_cam=[]
RawR_ves=[]
def format_matfile(path, name):
    a = sio.loadmat(path)[name]
    return dict(zip(a.dtype.names, [i for i in a.item()]))


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
colection=CoherenceSelect.CoherenceSelect()
for f in both_azimuth_files:
    if f[:-18] in colection:
        temp_file = format_matfile(files_r + 'SmoothData_' + f[:-15] + '.mat', 'CueConflictDataSmooth')
        RawR_trial.append(temp_file['resp_trial_conflict'])
        RawR_cam.append(temp_file['resp_vis'])
        RawR_ves.append(temp_file['resp_ves'])
SIZE=len(RawR_cam)
corr_list = [(np.triu(np.corrcoef([i.ravel() for i in RawR_trial[n]])).sum() - RawR_trial[n].shape[0]) / comb(
    RawR_trial[n].shape[0], 2) for n in range(SIZE)]
global selected_neurons
selected_neurons = [i for i in range(SIZE) if
                    corr_list[i] > 0.4]
RawR_cam=np.array(RawR_cam)[selected_neurons]
RawR_ves=np.array(RawR_ves)[selected_neurons]
RawR_cam_fit=[]
RawR_ves_fit=[]
for i in range(len(selected_neurons)):
    path = '../resultConsiderCoherence/vis/' + str(i) + '.xls'
    u=read(path)
    RawR_cam_fit.append(u)
for i in range(len(selected_neurons)):
    path = '../resultConsiderCoherence/ves/' + str(i) + '.xls'
    u=read(path)
    RawR_ves_fit.append(u)
# sum_cam=0
# sum_ves=0
# for k in range(len(selected_neurons)):
#     for i in range(9):
#         sum_cam+=RawR_cam[k][0][i]
sum_cam=sum(sum(sum(RawR_cam)))
# for k in range(len(selected_neurons)):
#     for i in range(9):
#         sum_ves+=RawR_ves[k][i][0]
sum_ves=sum(sum(sum(RawR_ves)))
ave_cam=sum_cam/(len(selected_neurons)*9)
ave_ves=sum_ves/(len(selected_neurons)*9)
SSR_cam=0
SST_cam=0
SSR_ves=0
SST_ves=0
SSE_cam=0
SSE_ves=0
for k in range(len(selected_neurons)):
    for i in range(9):
        SSE_cam=SSE_cam+(RawR_cam[k][0][i]-RawR_cam_fit[k][0][i])*(RawR_cam[k][0][i]-RawR_cam_fit[k][0][i])/(len(selected_neurons)*9)
        SSR_cam=SSR_cam+(RawR_cam_fit[k][0][i]-ave_cam)*(RawR_cam_fit[k][0][i]-ave_cam)/(len(selected_neurons)*9)
        SST_cam = SST_cam + (RawR_cam[k][0][i] - ave_cam) * (RawR_cam[k][0][i] - ave_cam) / (len(selected_neurons)*9)
for k in range(len(selected_neurons)):
    for i in range(9):
        SSE_ves = SSE_ves + (RawR_ves[k][i][0] - RawR_ves_fit[k][i][0]) * (RawR_ves[k][i][0] - RawR_ves_fit[k][i][0]) / (len(selected_neurons) * 9 * 9)
        SSR_ves = SSR_ves + (RawR_ves_fit[k][i][0] - ave_ves) * (RawR_ves_fit[k][i][0] - ave_ves) / (len(selected_neurons)*9)
        SST_ves=SST_ves+(RawR_ves[k][i][0]-ave_ves)*(RawR_ves[k][i][0]-ave_ves)/(len(selected_neurons)*9)
print(SSR_cam/SST_cam)
print(1-SSE_cam/SST_cam)
print(SSR_ves/SST_ves)
print(1-SSE_ves/SST_ves)