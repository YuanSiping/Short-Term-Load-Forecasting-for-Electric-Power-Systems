import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load raw data
df_raw = pd.read_csv('../data/load_30s_0514_0611.csv')
modeldata = df_raw[['TIME','A_CUR','B_CUR','C_CUR','T_ACT','C_ACT','B_ACT','A_ACT','S_DEM','T_DEM']]
modeldata.columns = ['Date','A_CUR','B_CUR','C_CUR','T_ACT','C_ACT','B_ACT','A_ACT','S_DEM','Load']

# Create temporal predictors
modeldata['Date'] = pd.to_datetime(modeldata.Date, format='%Y/%m/%d %H:%M:%S')

modeldata['Hour'] = pd.Series(
                    [modeldata.Date[idx].hour for idx in modeldata.index],
                    index = modeldata.index)
modeldata['Minute'] = pd.Series(
                    [modeldata.Date[idx].minute for idx in modeldata.index],
                    index = modeldata.index)
modeldata['Month'] = pd.Series(
                    [modeldata.Date[idx].month for idx in modeldata.index],
                    index = modeldata.index)

# Create suitable predictors:
modeldata['DayOfWeek'] = pd.Series(
                    [modeldata.Date[idx].isoweekday() for idx in modeldata.index],
                    index = modeldata.index)
np.savetxt('../data/week.csv', modeldata['DayOfWeek'], fmt="%d", header="week")
modeldata['isWeekend'] = pd.Series([int(modeldata.Date[idx].isoweekday() in [1,7]) for idx in modeldata.index],index = modeldata.index)

# Lagged predictors:
modeldata['PriorMinute_1'] = modeldata.Load.shift(1*2)
modeldata['PriorMinute_2'] = modeldata.Load.shift(2*2)
modeldata['PriorMinute_3'] = modeldata.Load.shift(3*2)
modeldata['PriorMinute_4'] = modeldata.Load.shift(4*2)
modeldata['PriorMinute_5'] = modeldata.Load.shift(5*2)
modeldata['PriorMinute_6'] = modeldata.Load.shift(6*2)
modeldata['PriorMinute_7'] = modeldata.Load.shift(7*2)
modeldata['PriorMinute_8'] = modeldata.Load.shift(8*2)
modeldata['PriorMinute_9'] = modeldata.Load.shift(9*2)
modeldata['PriorMinute_10'] = modeldata.Load.shift(10*2)
modeldata['PriorMinute_11'] = modeldata.Load.shift(11*2)
modeldata['PriorMinute_12'] = modeldata.Load.shift(12*2)
modeldata['PriorMinute_13'] = modeldata.Load.shift(13*2)
modeldata['PriorMinute_14'] = modeldata.Load.shift(14*2)
modeldata['PriorMinute_15'] = modeldata.Load.shift(15*2)
modeldata['PriorMinute_16'] = modeldata.Load.shift(16*2)
modeldata['PriorHour'] = modeldata.Load.shift(60*2)

# Drop NAN
modeldata = modeldata.dropna()

# Plot electric power load vs.time
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.plot_date(modeldata.Date, modeldata.Load,\
#             'b-',tz = None, xdate = True, ydate = False)
# # ax.set_title('Electric Power Demand Load for XG')
# ax.set_ylabel('Electric Power Demand Load, MW')

features=['Hour','Temperature','Minute','Month','DayOfWeek','isWeekend','A_CUR','B_CUR','C_CUR',\
          'A_VOL','B_VOL','C_VOL','T_ACT','C_ACT','B_ACT','A_ACT','S_DEM','F_DEM','Load',\
          'PriorMinute_1','PriorMinute_2','PriorMinute_3','PriorMinute_4',\
          'PriorMinute_5','PriorMinute_6','PriorMinute_7','PriorMinute_8',\
          'PriorMinute_9','PriorMinute_10','PriorMinute_11','PriorMinute_12',\
          'PriorMinute_13','PriorMinute_14','PriorMinute_15','PriorMinute_16','PriorHour']
X = modeldata[features]
Y = modeldata.Load

# fig = plt.figure()
# ax1 = fig.add_subplot(3, 1, 1)
# ax1.scatter(X.Hour, Y)
# ax1.set_xlabel("Hour")
# ax1.set_ylabel("Load")
# ax1.set_xlim([-1, 24.5])
#
# ax2 = fig.add_subplot(3, 1, 2)
# ax2.scatter(X.Minute, Y)
# ax2.set_xlim([-1, 60])
# ax2.set_xlabel("Minute")
# ax2.set_ylabel("Load")

# ax3 = fig.add_subplot(5, 1, 3)
# ax3.scatter(X.Month, Y)
# ax3.set_ylabel("Load")
# ax3.set_xlabel("Month")

# ax4 = fig.add_subplot(3, 1, 3)
# ax4.scatter(X.DayOfWeek, Y)
# ax4.set_ylabel("Load")
# ax4.set_xlabel("DayOfWeek")
#
# ax5 = fig.add_subplot(5, 1, 5)
# ax5.scatter(X.isWeekend, Y)
# ax5.set_ylabel("Load")
# ax5.set_xlabel("isWeekend")
# ax5.set_xlim([-0.5, 1.1])

#Scatter plots
# g = sns.PairGrid(modeldata, vars = ['Load', 'Temperature', 'Minute', 'Hour'])
# g = g.map_diag(plt.hist, edgecolor="w")
# g = g.map_offdiag(plt.scatter, edgecolor="w", s=20)

# fig = plt.figure()
# ax1 = fig.add_subplot(4, 4, 1)
# ax1.scatter(X.PriorHour, Y)
# ax1.set_xlabel("PriorHour")
# ax1.set_ylabel("Load")
#
# ax2 = fig.add_subplot(4, 4, 2)
# ax2.scatter(X.PriorMinute_15, Y)
# ax2.set_ylabel("Load")
# ax2.set_xlabel("Prior Minute_15 Load")
#
# ax3 = fig.add_subplot(4, 4, 3)
# ax3.scatter(X.PriorMinute_10, Y)
# ax3.set_ylabel("Load")
# ax3.set_xlabel("Prior Minute_10 Load")
#
# ax4 = fig.add_subplot(4, 4, 4)
# ax4.scatter(X.PriorMinute_1, Y)
# ax4.set_ylabel("Load")
# ax4.set_xlabel("Prior Minute_1 Load")
#
# ax5 = fig.add_subplot(4, 4, 5)
# ax5.scatter(X.A_CUR, Y)
# ax5.set_ylabel("Load")
# ax5.set_xlabel("A_CUR")
#
# ax6 = fig.add_subplot(4, 4, 6)
# ax6.scatter(X.B_CUR, Y)
# ax6.set_ylabel("Load")
# ax6.set_xlabel("B_CUR")
#
# ax7 = fig.add_subplot(4, 4, 7)
# ax7.scatter(X.C_CUR, Y)
# ax7.set_ylabel("Load")
# ax7.set_xlabel("C_CUR")
#
# ax8 = fig.add_subplot(4, 4, 8)
# ax8.scatter(X.A_VOL, Y)
# ax8.set_ylabel("Load")
# ax8.set_xlabel("A_VOL")
#
# ax9 = fig.add_subplot(4, 4, 9)
# ax9.scatter(X.B_VOL, Y)
# ax9.set_ylabel("Load")
# ax9.set_xlabel("B_VOL")
#
# ax10 = fig.add_subplot(4, 4, 10)
# ax10.scatter(X.C_VOL, Y)
# ax10.set_ylabel("Load")
# ax10.set_xlabel("C_VOL")
#
# ax11 = fig.add_subplot(4, 4, 11)
# ax11.scatter(X.T_ACT, Y)
# ax11.set_ylabel("Load")
# ax11.set_xlabel("T_ACT")
#
# ax12 = fig.add_subplot(4, 4, 12)
# ax12.scatter(X.A_ACT, Y)
# ax12.set_ylabel("Load")
# ax12.set_xlabel("A_ACT")
#
# ax13 = fig.add_subplot(4, 4, 13)
# ax13.scatter(X.B_ACT, Y)
# ax13.set_ylabel("Load")
# ax13.set_xlabel("B_ACT")
#
# ax14 = fig.add_subplot(4, 4, 14)
# ax14.scatter(X.C_ACT, Y)
# ax14.set_ylabel("Load")
# ax14.set_xlabel("C_ACT")
#
# ax15 = fig.add_subplot(4, 4, 15)
# ax15.scatter(X.S_DEM, Y)
# ax15.set_ylabel("Load")
# ax15.set_xlabel("S_DEM")
#
# ax16 = fig.add_subplot(4, 4, 16)
# ax16.scatter(X.Temperature, Y)
# ax16.set_ylabel("Load")
# ax16.set_xlabel("Temperature")

# plt.show()