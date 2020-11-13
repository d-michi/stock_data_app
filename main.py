import streamlit as st
import pandas as pd
import base64
import numpy as np
import talib as ta
from pandas_datareader import data
import matplotlib.pyplot as plt
from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')
import warnings
warnings.simplefilter('ignore')
import plotly.express as px
import plotly.io as pio
import yfinance as yf

#st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Web scraping of S&P 500 data
@st.cache
def load_data():
   url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
   html = pd.read_html(url, header = 0)
   df = html[0]
   return df

st.title('S&P 500 企業一覧')

df = load_data()
df

st.title('セクター')
st.text('0-資本財 1-ヘルスケア 2-情報技術 3-通信サービス 4-一般消費財 5-公益事業 6-金融 7-素材 8-不動産 9-生活必需品 10-エネルギー')

sector_unique = df['GICS Sector'].unique()
sector_unique

symbol_unique = df['Symbol'].unique()

len(sector_unique)

st.sidebar.header('銘柄検索')
sector = df.groupby('GICS Sector')

# Sidebar - Sector selection
sorted_sector_unique = sorted(df['GICS Sector'].unique())
selected_sector = st.sidebar.multiselect('Sector', sorted_sector_unique)

num_company = st.sidebar.slider('Number of Companies', 1, 10)


sorted_symbol_unique = sorted(df['Symbol'].unique())
selected_symbol = st.sidebar.multiselect('Symbol', sorted_symbol_unique)


# Filtering data
df_selected_sector = df[ (df['GICS Sector'].isin(selected_sector)) ]
df_selected_symbol = df[ (df['Symbol'].isin(selected_symbol)) ]


st.title('選択したセクター別の企業数')
st.write('銘柄数: ' + str(df_selected_sector.shape[0]))
st.dataframe(df_selected_sector)


#st.write(sector.first())

#st.title('選択したセクター別の企業数一覧')
#st.write(sector.describe())
#st.write(sector.get_group('セクター名'))

# Download S&P500 data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_sector), unsafe_allow_html=True)


data = yf.download(
        tickers = list(df_selected_sector.Symbol),
        period = "ytd", # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        interval = "1d", # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        group_by = 'ticker',
        auto_adjust = True,
        prepost = True,
        threads = True,
        proxy = None
    )

data2 = yf.download(
        tickers = list(df_selected_symbol.Symbol),
        period = "ytd", # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        interval = "1d", # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        group_by = 'ticker',
        auto_adjust = True,
        prepost = True,
        threads = True,
        proxy = None
    )



def price_plot(symbol):
  df = pd.DataFrame(data[symbol].Close)
  df['Date'] = df.index
  plt.fill_between(df.Date, df.Close, color='skyblue', alpha=0.3)
  plt.plot(df.Date, df.Close, color='skyblue', alpha=0.8)
  plt.xticks(rotation=90)
  plt.title(symbol, fontweight='bold')
  plt.xlabel('Date', fontweight='bold')
  plt.ylabel('Closing Price', fontweight='bold')
  return st.pyplot()

def price_plot2(symbol):
  df = pd.DataFrame(data2[symbol].Close)
  df['Date'] = df.index
  plt.fill_between(df.Date, df.Close, color='skyblue', alpha=0.3)
  plt.plot(df.Date, df.Close, color='skyblue', alpha=0.8)
  plt.xticks(rotation=90)
  plt.title(symbol, fontweight='bold')
  plt.xlabel('Date', fontweight='bold')
  plt.ylabel('Closing Price', fontweight='bold')
  return st.pyplot()



if st.button('チャート表示 個別企業'):
    st.header('Stock Closing Price')
    for i in list(df_selected_symbol.Symbol):
        price_plot2(i)



