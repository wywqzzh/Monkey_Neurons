#画拟合数据RE与RA图
import numpy as np
import scipy.io as sio
import os
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.special import comb
import xlrd
RawR = []  # firing curve, mean of neuron responses under different stimulus conditions
RawR_trial = []  # neuron responses under different stimulus conditions
RawR_cam=[]
RawR_ves=[]
pathx='./result/image'
import xlwt
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

def image1(RawR,RawR_cam,RawR_ves):
    RE=[]
    RA=[]
    for k in range(len(RawR)):
        Bi=-999999999999
        u1=float(max(RawR_cam[k][0]))
        u2=float(max(RawR_ves[k])[0])
        for i in range(len(RawR_cam[k][0])):
            for j in range(len(RawR_ves[k])):
                if float(RawR[k][i][j])>float(Bi):
                    Bi=float(RawR[k][i][j])
        re=(Bi-max(u1,u2))/(Bi+max(u1,u2))*100
        ra=(Bi-(u1+u2))/(Bi+(u1+u2))*100
        global RE1_m, RE1_d, RA1_m, RA1_d, RE2_m, RE2_d, RA2_m, RA2_d, RE3_m, RE3_d, RA3_m, RA3_d
        RE1_m.append(Bi)
        RE1_d.append(max(u1,u2))
        RA1_m.append(Bi)
        RA1_d.append((u1+u2))
        RE.append(re)
        RA.append(ra)
    global y
    y.append(RE)
    y.append(RA)
    plt.scatter(RE,RA,marker ='*',color='g')
    plt.xlabel('response enhancement')
    plt.ylabel('response additivity')
    path1=pathx+'/RE_RA_all.jpg'
    plt.savefig(path1, dpi=600)
    plt.show()
def image2(RawR,RawR_cam,RawR_ves):
    RE = []
    RA = []
    for k in range(len(RawR)):
        Bi =0
        u1=0
        for i in range(len(RawR_cam[k][0])):
            u1+=float(RawR_cam[k][0][i])
        u1 = u1/(float(len(RawR_cam[k][0])))
        u2 =0;
        # print(len(RawR_ves[k]))
        for i in range(len(RawR_ves[k])):
            u2+=float(RawR[k][i][0])
        u2=u2/(float(len(RawR_ves[k])))
        for i in range(len(RawR_cam[k][0])):
            for j in range(len(RawR_ves[k])):
                    Bi += float(RawR[k][i][j])
        Bi=Bi/((len(RawR_cam[k][0]))*(len(RawR_ves[k])));
        re = (Bi - max(u1, u2)) / (Bi + max(u1, u2)) * 100
        ra = (Bi - (u1 + u2)) / (Bi + (u1 + u2)) * 100;
        RE.append(re)
        RA.append(ra)
        global RE1_m, RE1_d, RA1_m, RA1_d, RE2_m, RE2_d, RA2_m, RA2_d, RE3_m, RE3_d, RA3_m, RA3_d
        RE2_m.append(Bi)
        RE2_d.append(max(u1, u2))
        RA2_m.append(Bi)
        RA2_d.append((u1 + u2))
    global y
    y.append(RE)
    y.append(RA)
    plt.scatter(RE, RA, marker='*',color='b')
    plt.xlabel('response enhancement')
    plt.ylabel('response additivity')
    path2 = pathx + '/RE_RA_vis.jpg'
    plt.savefig(path2, dpi=600)
    plt.show()
def image3(RawR,RawR_cam,RawR_ves):
    RE=np.zeros(len(RawR))
    RA = np.zeros(len(RawR))
    Re3_m=np.zeros(len(RawR))
    Re3_d=np.zeros(len(RawR))
    Ra3_m=np.zeros(len(RawR))
    Ra3_d=np.zeros(len(RawR))
    for i in range(9):
        for j in range(9):
            e = []
            e_m=[]
            e_d=[]
            a = []
            a_m=[]
            a_d=[]
            x = 1
            for k in range(len(RawR)):
                Bi = float(RawR[k][i][j])
                u1 = float(RawR_cam[k][0][i])
                u2 = float(RawR_ves[k][j][0])
                if (Bi + max(u1, u2)) == 0:
                    re=0
                else:
                    re=(Bi-max(u1,u2))/(Bi+max(u1,u2))*100
                if (Bi+(u1+u2))==0:
                    ra=0
                else:
                    ra=(Bi-(u1+u2))/(Bi+(u1+u2))*100
                e_m.append(Bi)
                e_d.append(max(u1, u2))
                a_m.append(Bi)
                a_d.append((u1+u2))
                e.append(re)
                a.append(ra)
            if x == 0:
                s = 0
            RE=RE+np.array(e)
            Re3_m = Re3_m + np.array(e_m)
            Re3_d=Re3_d+np.array(e_d)
            RA=RA+np.array(a)
            Ra3_m = Ra3_m + np.array(a_m)
            Ra3_d = Ra3_d + np.array(a_d)
    RE=RE/81
    Re3_m=Re3_m/81
    Re3_d = Re3_d / 81
    RA=RA/81
    Ra3_m = Ra3_m / 81
    Ra3_d = Ra3_d / 81
    global RE3_m,RE3_d,RA3_m,RA3_d
    RE3_m=Re3_m
    RE3_d=Re3_d
    RA3_m=Ra3_m
    RA3_d=Ra3_d
    global y
    y.append(RE)
    y.append(RA)
    plt.scatter(RE, RA, marker='*', color='r')
    plt.xlabel('response enhancement')
    plt.ylabel('response additivity')
    path3=pathx+'/RE_RA_ves.jpg'
    plt.savefig(path3, dpi=600)
    plt.show()
SIZE=99
for i in range(SIZE):
    path='./result/all/'+str(i)+'.xls'
    u=read(path)
    RawR.append(u)
for i in range(SIZE):
    path = './result/vis/' + str(i) + '.xls'
    u=read(path)
    RawR_cam.append(u)
for i in range(SIZE):
    path = './result/ves/' + str(i) + '.xls'
    u=read(path)
    RawR_ves.append(u)




global RE1_m,RE1_d,RA1_m,RA1_d,RE2_m,RE2_d,RA2_m,RA2_d,RE3_m,RE3_d,RA3_m,RA3_d
RE1_m=[]
RE1_d=[]
RA1_m=[]
RA1_d=[]

RE2_m=[]
RE2_d=[]
RA2_m=[]
RA2_d=[]


RE3_m=[]
RE3_d=[]
RA3_m=[]
RA3_d=[]

global y
y=[]
image1(RawR, RawR_cam, RawR_ves)
image2(RawR,RawR_cam,RawR_ves)
image3(RawR,RawR_cam,RawR_ves)
path='./result/RE_RA4.xls'
data_write(path,y)