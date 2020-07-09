import pandas as pd
import numpy as np
import os
import scipy.io as sio
import os
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.special import comb
import xlwt
# convert .mat file to dictionary
import matplotlib.pyplot as plt


def format_matfile(path, name):
    a = sio.loadmat(path)[name]
    return dict(zip(a.dtype.names, [i for i in a.item()]))


path_vis = './result/vis/'
path_all = './result/all/'
files_para = './data/ForNorData_MSTd/'
files_r = './data/AllData_MSTd/AllData/'
both_azimuth_files = os.listdir(files_para)
r_files = os.listdir(files_r)

VEST = list(range(0, 360 + 45, 45))
VIS = list(range(0, 360 + 45, 45))

global vest_a, vis_a, vest_e, vis_e, n1, n2, n3, e, c_vest, c_vis, n0, \
    S_vest_m, S_vis_m, RawR_trial, RawR
RawR_cam = []
RawR_ves = []
index = []
vest_a = []
vis_a = []
RawR = []
RawR_trial = []
RawR_cam = []
RawR_ves = []
index = []
Name = []
rate = 0.45
for f in both_azimuth_files:
    temp_file = format_matfile(files_para + f, 'ForNorData')
    vest_a.append(temp_file['vest_Direc'])
    vis_a.append(temp_file['vis_Direc'])
    index.append(files_r + f[:-15])
    temp_file = format_matfile(files_r + 'SmoothData_' + f[:-15] + '.mat', 'CueConflictDataSmooth')
    RawR_trial.append(temp_file['resp_trial_conflict'])
    RawR.append(temp_file['resp_conflict'])
    # RawR.append(temp_file['resp_conflict_200'])
    RawR_cam.append(temp_file['resp_vis'])
    RawR_ves.append(temp_file['resp_ves'])
    Name.append(f[:-18])
SIZE = len(RawR_trial)
corr_list = [(np.triu(np.corrcoef([i.ravel() for i in RawR_trial[n]])).sum() - RawR_trial[n].shape[0]) / comb(
    RawR_trial[n].shape[0], 2) for n in range(SIZE)]
selected_neurons = [i for i in range(SIZE) if
                    corr_list[i] > rate]

vest_a = [i + 360 if i < 0 else i for i in np.ravel(vest_a)]
vis_a = [i + 360 if i < 0 else i for i in np.ravel(vis_a)]
RawR_cam = np.array(RawR_cam)[selected_neurons]
RawR_ves = np.array(RawR_ves)[selected_neurons]
RawR = np.array(RawR)[selected_neurons]
# print(RawR.shape)
vest_a = np.array(vest_a)[selected_neurons]
vis_a = np.array(vis_a)[selected_neurons]
Name = list(np.array(Name)[selected_neurons])
VEST = list(range(-180, 180 + 45, 45))
# 细胞分类
evidence = pd.read_csv('./data/pAnova.dat', sep='\t')

FILE = list(np.array(evidence['FILE']))
p1comb = list(np.array(evidence[' p1comb']))
p1Vest = list(np.array(evidence['  p1Vest']))
p1Vis = list(np.array(evidence['p1Vis']))
pAnova2Veb = list(np.array(evidence[' pAnova2Veb']))
pVis = list(np.array(evidence[' pVis']))
pInteract = list(np.array(evidence[' pInteract']))
type = []
for i in range(len(FILE)):
    if p1Vis[i] <= 0.05:
        if p1Vest[i] <= 0.05:
            type.append(0)
        else:
            type.append(1)
    else:
        type.append(2)
type1 = []
for name in Name:
    index = FILE.index(name + 'ch1')
    type1.append(type[index])

listdir = os.listdir(path_all)

mean_combine_0 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
count_0 = 0
mean_combine_1 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
count_1 = 0
print(listdir)


for i in range(len(selected_neurons)):
    if(i==2):
        xxx=0
    all_path_file = os.path.join(path_all, str(i)+".xls")
    fit_all = pd.read_excel(all_path_file, encoding='gb18030', header=None)
    # Rcombine = RawR[i]

    # print(RawR_cam[i][0])
    VIS = list(RawR_cam[i][0])
    VIS.pop(8)
    MAX_VIS = max(VIS)
    position_vis = VIS.index(MAX_VIS)
    # position = position_vis

    VES = list(RawR_ves[i].T[0])
    VES.pop(8)
    MAX_VES = max(VES)
    position_ves = VES.index(MAX_VES)

    fit_all = np.array(fit_all)
    fit_all = list(fit_all[:, position_vis])
    # Rcombine = list(Rcombine[:, position_vis])
    fit_all.pop(8)
    # Rcombine.pop(8)

    # temp_ves = [0, 0, 0, 0, 0, 0, 0, 0]
    # temp_fit = [0, 0, 0, 0, 0, 0, 0, 0]
    # temp_Rcombine = [0, 0, 0, 0, 0, 0, 0, 0]
    # 峰值对齐
    # for j in range(8):
    #     temp_ves[(position_vis) % 8] = VES[(position_ves) % 8]
    #     temp_fit[(position_vis) % 8] = fit_all[(position_ves) % 8]
    #     temp_Rcombine[(position_vis) % 8] = Rcombine[(position_ves) % 8]
    #     position_vis += 1
    #     position_ves += 1
    #
    # fit_all = temp_fit
    # Rcombine = temp_Rcombine
    # VES = temp_ves
    index = 4
    temp_ves = [0, 0, 0, 0, 0, 0, 0, 0]
    temp_vis = [0, 0, 0, 0, 0, 0, 0, 0]
    temp_fit = [0, 0, 0, 0, 0, 0, 0, 0]
    # temp_Rcombine = [0, 0, 0, 0, 0, 0, 0, 0]
    # 峰值移动至中
    # print(position)
    for j in range(8):
        temp_ves[(index) % 8] = VES[(position_ves) % 8]
        # temp_vis[(index) % 8] = VIS[(position) % 8]
        temp_fit[(index) % 8] = fit_all[(position_ves) % 8]
        # temp_Rcombine[(index) % 8] = Rcombine[(position_ves) % 8]
        index += 1
        position_ves += 1
    # print(MAX_VIS)
    # print(temp_ves)
    # print(temp_fit)
    # print(max(fit_all)/MAX_VIS)

    # Rcombine = temp_Rcombine
    fit_all = temp_fit
    VIS = temp_vis
    VES = temp_ves
    # Rcombine.append(Rcombine[0])
    fit_all.append(fit_all[0])
    VIS.append(VIS[0])
    VES.append(VES[0])
    VES = list(VES)
    fit_all = list(fit_all)
    # print("fit_all/MAX_VIS:",)
    # Rcombine = list(Rcombine)
    if type1[i] == 0:
        count_0 += 1
        for k in range(len(mean_combine_0)):
            mean_combine_0[k]=mean_combine_0[k]+fit_all[k]/MAX_VIS
        print(i,MAX_VIS)
        print(VES)
        print(fit_all)
        print(fit_all / MAX_VIS)
        # mean_combine_0 = [(fit_all[j] / MAX_VIS + mean_combine_0[j]) for j in range(9)]
        # mean_combine_0 = [(Rcombine[j]/MAX_VIS + mean_combine_0[j]) for j in range(9)]

    elif type1[i] == 1:
        count_1 += 1
        for k in range(len(mean_combine_1)):
            mean_combine_1[k]=mean_combine_1[k]+fit_all[k]/MAX_VIS
            #a=a+b
        # mean_combine_1 = [(fit_all[j] / MAX_VIS + mean_combine_1[j]) for j in range(9)]
        # mean_combine_1 = [(Rcombine[j]/MAX_VIS + mean_combine_1[j]) for j in range(9)]
    # print(count_0)
    # print(mean_combine_0)
    # print(count_1)
    # print(mean_combine_1)
    plt.plot(VEST,VES)
    plt.plot(VEST,fit_all)
    plt.plot(VEST, [MAX_VIS, MAX_VIS, MAX_VIS, MAX_VIS, MAX_VIS, MAX_VIS, MAX_VIS, MAX_VIS, MAX_VIS])
    # plt.plot(VEST,[1,1,1,1,1,1,1,1,1])
    plt.legend(["realVes", "fitCombine", "maxVis"])
    plt.show()
print(count_0)
print(count_1)
mean_combine_0 = [mean_combine_0[i] / count_0 for i in range(9)]
mean_combine_1 = [mean_combine_1[i] / count_1 for i in range(9)]
print("mean_combine_0:")
print(mean_combine_0)
print("mean_combine_1:")
print(mean_combine_1)
# plt.plot(VEST,[KK,KK,KK,KK,KK,KK,KK,KK,KK],color='red')
plt.plot(VEST, [1, 1, 1, 1, 1, 1, 1, 1, 1], color='red')
plt.plot(VEST, mean_combine_0, color='black')
plt.plot(VEST, mean_combine_1, color='green')
plt.show()