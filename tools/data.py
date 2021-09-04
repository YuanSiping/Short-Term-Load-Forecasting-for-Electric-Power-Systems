import xlrd
import csv
import numpy as np
import pandas as pd

def save_data(data, filename):
    with open(filename, 'a', errors='ignore', newline='') as f:
        f_csv = csv.writer(f)
        #f_csv.writerow(["time", "load"])
        #f_csv.writerow(["time", "load","maxtemp","mintemp","avgtemp","humidity","precipitation"])
        f_csv.writerow(data)

# 将10s间隔负荷转换为1min平均负荷
def load_data(inf,outf):
    load = pd.read_csv(inf,header=0).values
    title = ['TIME','T_ACT','T_DEM','S_DEM','F_DEM']
    save_data(title, outf)
    for i in range(0,len(load)):
        if(i==0):
            T_ACT = 0
            T_DEM = 0
            S_DEM = 0
            F_DEM = 0
            num = 0
        if(i!=0 and str(load[i, 0])[:-3] != str(load[i-1, 0])[:-3]):
            T_ACT /= num
            T_DEM /= num
            S_DEM /= num
            F_DEM /= num
            time = str(load[i-1, 0])[:-3]
            loads = [time,float('%.2f' % T_ACT),float('%.2f' % T_DEM),float('%.2f' % S_DEM),float('%.2f' % F_DEM)]
            save_data(loads,outf)
            print(loads)
            T_ACT = 0
            T_DEM = 0
            S_DEM = 0
            F_DEM = 0
            num = 0
        T_ACT += load[i,1]
        T_DEM += load[i,2]
        S_DEM += load[i,3]
        F_DEM += load[i,4]
        num += 1

# 整合负荷数据与外部数据
def load_weather_data(lf,wf,of):
    load = pd.read_csv(lf, header=0).values
    weather = pd.read_csv(wf, header=0).values
    title = ['TIME', 'T_DEM', 'Temperature', 'weather_1', 'weather_2', 'weather_3']
    for h in range(0,24):
        title.append('hour_'+str(h))
    for m in range(1,13):
        title.append('month_'+str(m))
    print(title)
    save_data(title,of)
    j = 0
    for i in range(0, np.shape(load)[0]):
        time_load = str(load[i,0][0:11])
        if(time_load[-1] == ':'):
            time_load = time_load[:-1]
        time_weather = str(weather[j,0][0:11])
        print(time_load,time_weather)
        if (time_load != time_weather):
            j += 1
        weather_weather = weather[j,1] - 1
        w = [0,0,0]
        w[weather_weather] = 1
        hour = np.zeros(24)
        if(load[i,0][-5] == ' '):
            hour_load = int(load[i,0][-4])
        else:
            hour_load = int(load[i,0][-5:-3])
        hour[hour_load] = 1
        month = np.zeros(12)
        month_load = int(load[i,0][5])
        month[month_load-1] = 1
        data = [load[i, 0], load[i, 2], weather[j, 2], w[0], w[1], w[2]]
        for h in range(0, 24):
            data.append(hour[h])
        for m in range(0, 12):
            data.append(month[m])
        print(data)
        save_data(data, of)

# 将外部数据转换为标签数据
def load_calendar_data(lf,of):
    load = pd.read_csv(lf, header=0).values
    title = ['TIME','T_DEM','A_CUR','B_CUR','C_CUR','A_ACT','B_ACT','C_ACT','T_ACT','S_DEM']
    for m in range(1,13):
        title.append('month_'+str(m))
    for h in range(0,24):
        title.append('hour_'+str(h))
    for w in range(1,8):
        title.append('week_' + str(w))
    print(title)
    save_data(title,of)
    for i in range(0, np.shape(load)[0]):
        hour = np.zeros(24)
        hour[load[i,11]] = 1
        month = np.zeros(12)
        month[load[i,10]-1] = 1
        week = np.zeros(7)
        week[load[i,12]-1] = 1
        data = []
        for j in range(0,10):
            data.append(load[i,j])
        for m in range(0, 12):
            data.append(month[m])
        for h in range(0, 24):
            data.append(hour[h])
        for w in range(0,7):
            data.append(week[w])
        save_data(data,of)

# 将10s间隔数据抽取为1min间隔数据
def extract_to_one_min(inf,outf):
    data = pd.read_csv(inf).values
    title = data[0]
    save_data(title,outf)
    i = 1
    while(i < len(data)):
        data_row = data[i]
        i += 6
        save_data(data_row,outf)

# 将10s间隔数据抽取为30s间隔数据
def extract_to_thirty_sec(inf,outf):
    data = pd.read_csv(inf).values
    title = data[0]
    save_data(title,outf)
    i = 1
    while(i < len(data)):
        data_row = data[i]
        i += 3
        save_data(data_row,outf)

# 将15s间隔数据抽取为30s间隔数据
def extract_to_thirty_sec2(inf, outf):
    data = pd.read_csv(inf).values
    title = data[0]
    save_data(title, outf)
    i = 1
    while (i < len(data)):
        data_row = data[i]
        i += 2
        save_data(data_row, outf)

# 将外部数据转换为标签数据
def load_calendar_data_(lf,of):
    load = pd.read_csv(lf, header=0).values
    title = []
    for w in range(1,8):
        title.append('week_' + str(w))
    print(title)
    save_data(title,of)
    for i in range(0, np.shape(load)[0]):
        week = np.zeros(7)
        week[load[i]-1] = 1
        data = []
        for w in range(0,7):
            data.append(week[w])
        save_data(data,of)

if __name__ == '__main__':
    # intput_files_name = '../data/data50_0613.csv'
    # output_files_name = '../data/data50_0613_1min.csv'
    # load_data(intput_files_name,output_files_name)
    # load_files_name = '../data/data50_10s_0514_0611.csv'
    # weather_files_name = '../data/weather_0514_0611.csv'
    # output_files_name = '../data/load_weather_10s.csv'
    # load_weather_data(load_files_name, weather_files_name, output_files_name)
    intput_files_name = '../data/30_0719_0809.csv'
    output_files_name = '../data/30_0719_0809_.csv'
    load_calendar_data(intput_files_name,output_files_name)