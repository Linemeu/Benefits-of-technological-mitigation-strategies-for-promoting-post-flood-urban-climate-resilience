# 计算多元回归模型的MA
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from vmdpy import VMD
import pandas as pd
from scipy.fftpack import fft

wurenji_hour = np.load('wurenji_hour.npy', allow_pickle=True)
gaode_hour = np.load('gaode_hour.npy', allow_pickle=True)
wendang_hour = np.load('wendang_hour.npy', allow_pickle=True)
longxishui_hour = np.load('longxishui_hour.npy', allow_pickle=True)
zhihuishuiwu_hour = np.load('zhihuishuiwu_hour.npy', allow_pickle=True)
TPI = np.load('TPI_hour.npy', allow_pickle=True)
TPI=np.log(TPI+1)
WEPI = np.load('WPI_hour.npy', allow_pickle=True)
WEPI=np.log(WEPI+1)
CPI = np.load('CPI_hour.npy', allow_pickle=True)
CPI=np.log(CPI+1)
BPI = np.load('BPI_hour.npy', allow_pickle=True)
# BPI=np.log(BPI+1)
CAPI = np.load('CAPI_hour.npy', allow_pickle=True)
CAPI=np.log(CAPI+1)
rain = np.load('maxrain.npy', allow_pickle=True)

REE = np.load('MR_result/REE_hour.npy', allow_pickle=True)


alpha = 9000  # moderate bandwidth constraint
tau = 0.000001  # noise-tolerance (no strict fidelity enforcement)
K = 2  # 3 modes
DC = 0  # no DC part imposed
init = 1  # initialize omegas uniformly
tol = 1e-7
name = ['TPI_hour', 'WEPI_hour', 'CPI_hour', 'BPI_hour',  'CAPI_hour', \
        'wurenji_hour', 'gaode_hour', 'wendang_hour', 'longxishui_hour', 'zhihuishuiwu_hour', 'rain', 'REE']
# plt.subplots(5,3,dpi=320)
for i, f in enumerate([TPI, WEPI, CPI, BPI,CAPI, \
                       wurenji_hour, gaode_hour, wendang_hour, longxishui_hour, zhihuishuiwu_hour, rain, REE]):
    u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)
    np.save(name[i] + "_vmd.npy", u[0])


import numpy as np
import pandas as pd
# 接下来整理变量
# 1.感知指数；2.降雨数据 3.新技术指数
TPI = np.load('TPI_hour_vmd.npy', allow_pickle=True)
WPI = np.load('WEPI_hour_vmd.npy', allow_pickle=True)
CAPI = np.load('CAPI_hour_vmd.npy', allow_pickle=True)
BPI = np.load('BPI_hour_vmd.npy', allow_pickle=True)
CPI = np.load('CPI_hour_vmd.npy', allow_pickle=True)

rain = np.load('maxrain.npy', allow_pickle=True)
# rain=np.load('rain_vmd.npy',allow_pickle=True)
wurenji_hour = np.load('wurenji_hour_vmd.npy', allow_pickle=True)
gaode_hour = np.load('gaode_hour_vmd.npy', allow_pickle=True)
wendang_hour = np.load('wendang_hour_vmd.npy', allow_pickle=True)
longxishui_hour = np.load('longxishui_hour_vmd.npy', allow_pickle=True)
zhihuishuiwu_hour = np.load('zhihuishuiwu_hour_vmd.npy', allow_pickle=True)
REE = np.load('REE_vmd.npy', allow_pickle=True)

df = pd.DataFrame()

# 1.对EPI进行ADL建模
df = pd.DataFrame()
df['REE'] = REE

df['TPI'] = TPI
df['d_TPI'] = df['TPI'].diff(periods=1)

df['CPI'] = CPI
df['d_CPI'] = df['CPI'].diff(periods=1)

df['BPI'] = BPI
# df['BPI']=np.log(BPI+1)
df['d_BPI'] = df['BPI'].diff(periods=1)

df['WPI'] = WPI
# df['WEPI']=np.log(WEPI+1)
df['d_WPI'] = df['WPI'].diff(periods=1)


df['CAPI'] = CAPI
# df['CAPI']=np.log(CAPI+1)
df['d_CAPI'] = df['CAPI'].diff(periods=1)



df['BPI_l1'] = df['BPI'].shift(-1)
df['TPI_l1'] = df['TPI'].shift(-1)
df['WPI_l1'] = df['WPI'].shift(-1)

df['rain'] = rain
account_t = 2
account_t2 = 6

#画图分析
#REE分析
import matplotlib.pyplot as plt
REE_hour=REE
fig = plt.figure(figsize=(15, 9))

# 设置图例并且设置图例的字体及大小
font1 = {'family': 'Arial',
         'weight': 'semibold',
         'size': 25,
         }
title_font = {'family': 'Arial',
         'weight': 'semibold',
         'size': 30,
         }

ax1 = fig.add_subplot(111)
ax1.plot(range(len(REE)), np.array(REE), '-', color='red', label='REE')
ax1.set_xlabel('Hours',fontdict=font1)
# ax1.plot(df['ft_5min'], df['density'], '-', color='r',label='平均密度变化')
ax1.set_ylabel('CAPI',fontdict=font1)
ax1.legend(loc=2,prop=title_font)

plt.show()
plt.savefig('.\\MR_fig\\REE_ts2.jpg')








# -----------------------REE----------------------------------------------
# df['REE'] = np.exp(REE/100)
df['REE']=np.log(df['REE']+1)

#河南省应急救援协会于7月20日17时启动Ⅰ级应急响应，同时成立“7·20河南洪涝灾害”救灾指挥部，由协会党支部、工会、安专委、会员单位、29支应急救援队伍、数千名志愿者组成的救灾体系，在指挥部的统一协调指挥下，一场同自然灾害抗争、守卫家园的大规模救灾行动拉开了帷幕。
#一级响应 7.20 17，209h
#从7月17日开始，河南西部、中西部地区出现持续性强降水。郑州昨天（7月20日）24小时降雨量达到624.1毫米。其中，20日16时至17时，一小时降雨量达到201.9毫米，日降雨量、小时雨强均突破

#2021年7月20日，针对河南省防汛抢险救灾工作，国家防总启动防汛Ⅲ级应急响应，216
#2021年7月20日08时至7月21日06时，216-24+8=200

ree_T = []
for i in df.index:
    e = 1
    if i >=200:
        e = 1
    else:
        e = 0
    ree_T.append(e)
df['ree_T'] = ree_T
df['REE'] = df['REE'] * df['ree_T']




delay_Ir = []
for i, e in enumerate(df['REE']):
    alpha = 0.8
    f = np.array([np.power(alpha, i - j) for j in range(i + 1)])
    I = df['REE'][0:i + 1]
    Ir = sum(list(I * f)[-account_t:])
    delay_Ir.append(Ir)
df['dalay_REE'] = delay_Ir
# df['wurenji']=np.log(df['dalay_wurenji']+1)
df['REE'] = df['dalay_REE']





# -----------------------无人机----------------------------------------------



df['wurenji'] = np.exp(wurenji_hour/100)
df['wurenji']=np.log(df['wurenji']+1)



wurenji_T = []
for i in df.index:
    e = 1
    if i >= 240:
        e = 1
    else:
        e = 0
    wurenji_T.append(e)
df['wurenji_T'] = wurenji_T
df['wurenji'] = df['wurenji'] * df['wurenji_T']



delay_Ir = []
for i, e in enumerate(df['wurenji']):
    alpha = 0.8
    f = np.array([np.power(alpha, i - j) for j in range(i + 1)])
    I = df['wurenji'][0:i + 1]
    Ir = sum(list(I * f)[-account_t:])
    delay_Ir.append(Ir)
df['dalay_wurenji'] = delay_Ir
# df['wurenji']=np.log(df['dalay_wurenji']+1)
df['UAV'] = df['dalay_wurenji']





# ---------------------文档-------------------------------------------------
# 7月20日21时许
# 7月20日21时许，《待救援人员信息》在线表格创建后率先在朋友圈传播，而后在社交网络平台、新闻媒体等平台引发大量关注
# #213
lsd_T = []
for i in df.index:

    e = 1
    # if i > 240:
    if i >= 213:
        e = 1
    else:
        e = 0
    lsd_T.append(e)
df['lsd_T'] = lsd_T
df['wendang'] = wendang_hour
# df['wendang']=np.log(df['wendang']+1)
# df['wendang']=np.log(df['wendang']+1)
df['wendang'] = df['wendang'] * df['lsd_T']

df['wendang'] = np.exp(df['wendang']/100)
df['wendang']=np.log(df['wendang']+1)
#

delay_Ir = []
for i, e in enumerate(df['wendang']):
    alpha = 0.8
    f = np.array([np.power(alpha, i - j) for j in range(i + 1)])
    I = df['wendang'][0:i + 1]
    Ir = sum(list(I * f)[-account_t:])
    delay_Ir.append(Ir)
df['dalay_wendang'] = delay_Ir
df['LSD'] = df['dalay_wendang']


# -----------------高德-----------------------------------------------------
df['gaode'] = gaode_hour
df['gaode'] = np.exp(df['gaode']/100)
df['gaode']=np.log(df['gaode']+1)

#7 月 21 日消息 今日凌晨 @高德地图 官方微博称已经上线河南暴雨信息互助通道。在河南当地的用户
amap_T = []
for i in df.index:
    e = 1
    if i >= 216:
        e = 1
    else:
        e = 0
    amap_T.append(e)
df['amap_T'] = amap_T

df['gaode'] = df['gaode'] * df['amap_T']



delay_Ir = []
for i, e in enumerate(df['gaode']):
    alpha = 0.8
    f = np.array([np.power(alpha, i - j) for j in range(i + 1)])
    I = df['gaode'][0:i + 1]
    Ir = sum(list(I * f)[-account_t:])
    delay_Ir.append(Ir)
df['dalay_gaode'] = delay_Ir
df['AMap'] = df['dalay_gaode']

# ----------------------龙吸水-----------------------------------------

df['longxishui'] = longxishui_hour

longxishui_T = []
for i in df.index:
    # 月22日凌晨4时，根据北京市统一部署，北京排水集团紧急组建抢险突击队驰援郑州参与抢险工作，包括2台“龙吸水”，4
    # 组大型抽排单元、16
    e = 1

    if i >=244:
        e = 1
    else:
        e = 0
    longxishui_T.append(e)
df['longxishui_T'] = longxishui_T
df['longxishui'] = longxishui_hour
df['longxishui']=df['longxishui']*df['longxishui_T']

# df['longxishui'] = np.exp(df['longxishui']/100)
df['longxishui'] = np.exp(df['longxishui']/100)
df['longxishui']=np.log(df['longxishui']+1)

# df['longxishui']=np.log(df['longxishui']+1)
delay_Ir = []
for i, e in enumerate(df['longxishui']):
    alpha = 0.8
    f = np.array([np.power(alpha, i - j) for j in range(i + 1)])
    I = df['longxishui'][0:i + 1]
    Ir = sum(list(I * f)[-account_t:])
    delay_Ir.append(Ir)
df['dalay_longxishui'] = delay_Ir
# df['longxishui']=df['dalay_longxishui']
df['DW'] = df['dalay_longxishui']



rain_T = []
for i in df.index:
    e = 1
    rain_T.append(e)
# -------------------rain-------------------------------------------------------------------------------------
df['rain'] = rain

df['Rain'] = df['rain']
df['rain_T'] = rain_T
df['rain'] = df['rain'] * df['rain_T']
# df['rain']=(df['rain']-df['rain'].mean())/df['rain'].std()
df['rain_l6'] = df['rain'].shift(6)
delay_Ir = []
for i, e in enumerate(df['rain_l6']):
    alpha = 0.8
    f = np.array([np.power(alpha, i - j) for j in range(i + 1)])
    I = df['rain_l6'][0:i + 1]
    Ir = sum(list(I * f)[-account_t2:])
    delay_Ir.append(Ir)
df['dalay_rain'] = delay_Ir
df['Rain'] = df['dalay_rain']

# ---------------------------------------交通--------------------------------------------------
import semopy
import pandas as pd


df = df.apply(lambda x: (x - np.mean(x)) / (np.std(x)))


# df['REE']=np.log(df['REE']+1)
df['UAV_l1'] = df['UAV'].shift(-1)
df['AMap_l1'] = df['AMap'].shift(-1)
df['DW_l1'] = df['DW'].shift(-1)
df['LSD_l1'] = df['LSD'].shift(-1)
df['CPI_l1'] = df['CPI'].shift(-1)
df['CAPI_l1'] = df['CAPI'].shift(-1)
df['BPI_l1'] = df['BPI'].shift(-1)
df['TPI_l1'] = df['TPI'].shift(-1)
df['BPI_l1'] = df['BPI'].shift(-1)

df['REE_l1'] = df['REE'].shift(-1)



