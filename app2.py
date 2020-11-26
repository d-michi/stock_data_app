import pandas as pd
import talib as ta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import pandas_datareader as pdr

def candle_patt(df,x,all_fig,params):
    '''ローソク足パターンのローソク足チャートへの描画'''
    
    # パターンチェック & シグナル値を描画用に置き換え
    df['Marubozu'] = ta.CDLMARUBOZU(df['Open'],df['High'],df['Low'],df['Close']) * df['High'] / 100
    df['Engulfing_Pattern'] = ta.CDLENGULFING(df['Open'],df['High'],df['Low'],df['High']) * df['Close'] / 100
    df['Hammer'] = ta.CDLHAMMER(df['Open'],df['High'],df['Low'],df['Close']) * df['High'] / 100
    df['Dragonfly_Doji'] = ta.CDLDRAGONFLYDOJI(df['Open'],df['High'],df['Low'],df['Close']) * df['High'] / 100    
    
    # 列名を作成
    pattern_list = list(df.loc[:,'Marubozu':'Dragonfly_Doji'].columns)
    label_list = [ k+'_label' for k in list(df.loc[:,'Marubozu':'Dragonfly_Doji'].columns)]
    
    # 0をNaNで埋める
    df[pattern_list] = df[pattern_list].where(~(df[pattern_list] == 0.0), np.nan)
        
    # 売り買いラベルの作成
    df[label_list] = df[pattern_list]
    df[label_list] = df[label_list].where(~(df[label_list] > 0), 1)
    df[label_list] = df[label_list].where(~(df[label_list] < 0), -1)
    df[label_list] = df[label_list].where(~(df[label_list] == 1), '買い')
    df[label_list] = df[label_list].where(~(df[label_list] == -1), '売り')
    
    # 発生価格の絶対値化
    df[pattern_list] = df[pattern_list].abs()
    
    # 各シグナルを描画
    for pattern in list(df.loc[:,'Marubozu':'Dragonfly_Doji'].columns):
        all_fig.add_trace(go.Scatter(x=x, y=df[pattern],mode='markers+text',text=df[label_list],textposition="top center",name=pattern,
                                    marker = dict(size = 9),opacity=0.8),row=1, col=1)
    
    return all_fig


# 各指標の描画用関数
def call_indi(df,x,all_fig,params):
    '''テクニカル分析の描画'''
    
    # 移動平均線
    if 'SMA' in params.keys(): 
        # 移動平均線の計算と描画
        for day in params['SMA']:
            df[str(day) + 'D_SMA'] = ta.SMA(df["Close"], timeperiod=day) # 計算
            all_fig.add_trace(go.Scatter(x=x, y=df[str(day) + 'D_SMA'],mode='lines',name=str(day) + 'D_SMA'),row=1, col=1) # 描画
        
    # MACD
    if 'MACD' in params.keys():
        df['MACD'],df['MACDSIGNAL'],_ = ta.MACD(pdr['Close'], fastperiod=params['MACD']['short'], 
                                                slowperiod=params['MACD']['long'], signalperiod=9)# 計算

        all_fig.add_trace(go.Scatter(x=x, y=df['MACD'],name='MACD',mode='lines'),row=3, col=1) # MACD 描画
        all_fig.add_trace(go.Scatter(x=x, y=df['MACDSIGNAL'],name='SIGNAL',mode='lines'),row=3, col=1) # MACD シグナル描画
        
    # ストキャスティクス
    if 'ストキャスティクス' in params.keys():
        STOCH = params['ストキャスティクス']
        df['stoch_k'], df['stoch_d'] = ta.STOCH(df['High'],df['Low'],df['Close'], 
                                           fastk_period=STOCH['fastk_period'], slowk_period=STOCH['slowk_periodg'], slowk_matype=0, slowd_period=STOCH['slowd_period'], slowd_matype=0)

        all_fig.add_trace(go.Scatter(x=x, y=df['stoch_k'],name='%K',mode='lines'),row=4, col=1) # Slow％K 描画
        all_fig.add_trace(go.Scatter(x=x, y=df['stoch_d'],name='%D',mode='lines'),row=4, col=1) # Slow％D 描画    
                          
    return all_fig

def plot_chart(df,params):
    
    # 期間のデータの加工
    long_day = pd.Series(data=np.nan,index=pd.date_range(start=df.index[-1], end=df.index[-1] + datetime.timedelta(days=20), freq='D'),name='add_days')
    df = pd.merge(df, long_day, left_index=True, right_index=True,how='outer').drop(['add_days'],axis=1)
    
    # 土日でデータのスペースが空くのを補正するための設定
    x = np.arange(df.shape[0])
    interval = 20
    vals = np.arange(df.shape[0],step=interval)
    labels = list(df[::interval].index)
    
    # ローソク足チャート部分
    trace1 = go.Candlestick(
                    x=x,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close']
                    )
    
    # 複数グラフの作図設定:4行1列
    all_fig = make_subplots(rows=4, cols=1,shared_xaxes=True,vertical_spacing=0.08,row_heights=[0.4,0.1,0.2,0.2])
    
    all_fig.add_trace(trace1,row=1, col=1)

    # 指標の描画
    all_fig = call_indi(df,x,all_fig,params)
    
    # ローソクパターンの描画
    all_fig = candle_patt(df,x,all_fig,params)
    
    # 描画の設定
    all_fig.update_xaxes(tickvals=vals,ticktext=labels,tickangle=-45, row=4, col=1) #軸の値
    all_fig.update_yaxes(range=[0, 100],dtick=20, row=4, col=1) #y軸の値
    all_fig.update_layout(width=1000,height=1000)
    
    all_fig.show()

#-----------------------------
# 株データの取得
pdr =pdr.stooq.StooqDailyReader(symbols='6701.jp', start='JAN-01-2019', end="JUL-3-2020").read().sort_values(by='Date',ascending=True)

# テクニカル分析のパラメータ設定
params = {
         'SMA':[5,25,50],
         'MACD':{'short':12,'long':26},
         'ストキャスティクス':{'fastk_period':14,'slowk_periodg':3,'slowd_period':3},
         }

# 描画関数の呼び出し
plot_chart(pdr,params)