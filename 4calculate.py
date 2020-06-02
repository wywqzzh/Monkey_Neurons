#计算bimodal的拟合数据
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


def read():
    book = xlrd.open_workbook('./result/参数4_nobound_bimodal.xls')
    sheet1 = book.sheets()[0]
    nrows = sheet1.nrows
    ncols = sheet1.ncols
    current_vect=[]
    for i in range(nrows):
        for j in range(ncols):
            current_vect.append(sheet1.cell(i,j).value)
    return current_vect

def spherical_sinusoid(az, az0, ele, ele0, c, n):
    # N x3 prefered heading vector in Cartesian coordinate
    H_hat = [(np.cos(ele0)*np.cos(az0)), np.cos(ele0)*np.sin(az0), np.sin(ele0)]
    # 1x3 stimulus heading vector
    H = [(np.cos(ele)*np.cos(az)), np.cos(ele)*np.sin(az), np.sin(ele)]

    big_phi = np.arccos(np.matrix(H)*np.matrix(H_hat))
    return c*(np.array((1 + np.cos(big_phi))/2)**n)

# convert .mat file to dictionary
def format_matfile(path, name):
    a = sio.loadmat(path)[name]
    return dict(zip(a.dtype.names, [i for i in a.item()]))



def expand_dimension(v):
    return np.array([i*np.ones((9,9)) for i in v])



def nordiv_Bin(current_vect):
    global vest_a, vis_a, vest_e, vis_e, n1, n2, n3, e, c_vest, c_vis, n0,\
                S_vest_m, S_vis_m
#     current_vect[:, cnt] = vect
    [baselineConst,  alpha,  d_ves,  d_vis,  _] = current_vect
    L_neuron = expand_dimension(d_ves)*S_vest_m + expand_dimension(d_vis)*S_vis_m + expand_dimension(baselineConst) # S_vest_m 9*9*200, d_ves 1*200, baselineConst 1*200
    return L_neuron

# def generate_RawR():
#     l = []
#     for n in range(200):
#         pick = np.random.randint(RawR_trial[n].shape[0])
#         l.append(RawR_trial[n][pick])
#     return l


def calculate_R(current_vect):
    current_vect = current_vect.reshape(5,len(selected_neurons))#.repeat(trials, axis=1)
    L_neuron = nordiv_Bin(current_vect)
    normPool = e*np.mean(L_neuron**n3)
    alpha = current_vect[1]
    # Rmax_m = np.array([m*np.full((9,9),1) for m in current_vect[4]])
    Rs = Rmax_m*(L_neuron**n3)/(expand_dimension(alpha) + normPool)
    return Rs

def data_write(file_path, datas):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet

    # 将数据写入第 i 行，第 j 列
    i = 0
    for data in datas:
        for j in range(len(data)):
            sheet1.write(i, j, data[j])
        i = i + 1

    f.save(file_path)  # 保存文件

SIZE=158
def init():
    files_para = './data/ForNorData_MSTd/'
    files_r = './data/AllData_MSTd/AllData/'
    both_azimuth_files = os.listdir(files_para)

    r_files = os.listdir(files_r)

    VEST = list(range(0, 360 + 45, 45))  # H: vest azimuth
    VIS = list(range(0, 360 + 45, 45))  # H: vis azimuth

    # H_hat: vest azimuth load, vis azimuth load, vest elevation all zero, vis elevation all zero,
    # H: vest azimuth in [0,360,45] range, vis azimuth in [0,360,45] range,
    # vest elevation all zero, vis elevation all zero,
    global vest_a, vis_a, vest_e, vis_e, n1, n2, n3, e, c_vest, c_vis, n0, \
        S_vest_m, S_vis_m,RawR_trial,RawR
    vest_a = []  # vest azimuth in H_hat
    vis_a = []  # vis azimuth in H_hat
    RawR = []  # firing curve, mean of neuron responses under different stimulus conditions, 就是我们最后要fit的y值（平均数）
    RawR_trial = []  # neuron responses under different stimulus conditions, 就是我们最后要fit的y值（实际trial的y值）
    vest_resp = [] # 不需要用到，只是在算Rmax的时候用了
    vis_resp = []
    for f in both_azimuth_files:
        temp_file = format_matfile(files_para + f, 'ForNorData')
        vest_a.append(temp_file['vest_Direc'])
        vis_a.append(temp_file['vis_Direc'])

        temp_file = format_matfile(files_r + 'SmoothData_' + f[:-15] + '.mat', 'CueConflictDataSmooth')
        RawR_trial.append(temp_file['resp_trial_conflict'])
        RawR.append(temp_file['resp_conflict'])
        vest_resp.append(temp_file['resp_ves'])
        vis_resp.append(temp_file['resp_vis'])
        # temp_file = format_matfile(files_r + f[:-15] + '.mat', 'CueConflictData')
        # RawR_trial.append(temp_file['resp_trial_conflict_200'])
        # RawR.append(temp_file['resp_conflict_200'])
    vest_a = [i + 360 if i < 0 else i for i in np.ravel(vest_a)]
    vis_a = [i + 360 if i < 0 else i for i in np.ravel(vis_a)]

    vest_e = np.zeros(len(vest_a))
    vis_e = np.zeros(len(vis_a))

    vest_stim_ele = 0
    vis_stim_ele = 0

    n1 = n2 = n3 = e = c_vest = c_vis = 1
    n0 = 2

    init_vect = np.array([np.random.rand(SIZE) + 9.5, np.random.rand(SIZE) + 0.5,
                          np.random.rand(SIZE) - 0.53, np.random.rand(SIZE) + 0.55,
                          np.random.rand(SIZE)])  # 参数初始值
    global Rmax_m
    Rmax_m = np.zeros((SIZE, 9, 9))
    S_vest_m = np.zeros((SIZE, 9, 9))
    S_vis_m = np.zeros((SIZE, 9, 9))
    for k in range(len(VEST)):
        for j in range(len(VIS)):
            vest_stim_az = VEST[k]
            vis_stim_az = VIS[j]
            Rmax = np.maximum(np.ravel(np.array(vest_resp)[:, k]), np.ravel(np.array(vis_resp)[:, :, j]))
            Rmax_m[:, k, j] = Rmax
            S_vest = spherical_sinusoid(vest_stim_az, vest_a, vest_stim_ele, vest_e, c_vest, n0)
            S_vest_m[:, k, j] = S_vest
            S_vis = spherical_sinusoid(vis_stim_az, vis_a, vis_stim_ele, vis_e, c_vis, n0)
            S_vis_m[:, k, j] = S_vis


    corr_list = [(np.triu(np.corrcoef([i.ravel() for i in RawR_trial[n]])).sum() - RawR_trial[n].shape[0]) / comb(
        RawR_trial[n].shape[0], 2) for n in range(SIZE)]
    global selected_neurons
    selected_neurons = [i for i in range(SIZE) if
                        corr_list[i] > 0.4]  # select good neurons: correlation between trials over 0.4
    # global  selected_neurons
    # selected_neurons=list(range(200))
    S_vest_m=S_vest_m[selected_neurons]
    S_vis_m = S_vis_m[selected_neurons]
    Rmax_m=Rmax_m[selected_neurons]
    init_vect = init_vect[:, selected_neurons]
    vis_a = np.array(vis_a)[selected_neurons].tolist()
    vest_a = np.array(vest_a)[selected_neurons].tolist()
    RawR = np.array(RawR)[selected_neurons]
    RawR_trial = np.array(RawR_trial)[selected_neurons]
    S_vest_m = S_vest_m
    S_vis_m = S_vis_m
    current_vect=np.array(read())


    rs=calculate_R(current_vect)
    for i in range(len(selected_neurons)):
        path='./result/all/'+str(i)+'.xls'
        data_write(path,rs[i])

init()