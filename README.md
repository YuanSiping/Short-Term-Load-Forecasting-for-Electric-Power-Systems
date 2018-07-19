# Short-Term-Load-Forecasting-for-Electric-Power-Systems
### ENTSO-E : Inputing the ENTSO-E Hourly Load.
Models|NRMSE|MAE|MAPE
------------- | -------------| -------------| -------------
HMM | 0.255 | 1058.75 | 0.148
ARIMA | 0.198 | 807.97 | 0.108
DWT-ARIMA | 0.0805 | 565.91 | 0.0876
SVR | 0.0409 | 146.80 | 0.0210
GPR | 0.0435 | 162.34 | 0.0232
FFNN | 0.0504 | 200.59 | 0.0282
Clustering | 0.0684 | 271.51 | 0.0384
LSTM | 0.0451 | 167.85 | 0.0239
Seq2Seq | 0.0424 | 153.74 | 0.0219
DBN | 0.0434 | 162.38 | 0.0232
RFR | 0.0411 | 154.94 | 0.0221
GDRT | 0.0424 | 157.87| 0.0225
XGBoost | 0.0418 | 154.14 | 0.0219
### ENTSO-E & NCEI ISD : Inputing the ENTSO-E Hourly Load, Weather and Calendar.
Models|NRMSE|MAE|MAPE
------------- | -------------| -------------| -------------
SVR | 0.0390 | 146.30 | 0.0209
LSTM | 0.0411 | 155.34 | 0.0218
DBN | 0.0398 | 142.04 | 0.0205
Seq2Seq | 0.0417 | 156.29 | 0.0225
LSTM-Seq2Seq | 0.0376 | 137.02 | 0.0195
