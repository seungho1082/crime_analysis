#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install folium
#!pip install xgboost
#!pip install lightgbm
#!pip install tensorflow


# In[2]:


import pandas as pd
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import folium
import requests
from bs4 import BeautifulSoup
from glob import glob
from tqdm.notebook import tqdm
import os
from folium import plugins
import statsmodels.tsa.api as tsa
import itertools
import statsmodels.api as sm
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import StandardScaler


# In[3]:


# 그래프를 출력할 때 한글 글씨체 사용
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)


# # 데이터 전처리
# 
# - 인덱스 지정
# - 열 추가 (합계)

# ## 지역별 가로등 개수

# ### 경기도 가로등 개수

# In[4]:


Gyeonggi_lamp = pd.read_csv('./경기_가로등현황.csv', encoding='euc-kr')
Gyeonggi_lamp2 = Gyeonggi_lamp
Gyeonggi_lamp2


# In[5]:


# 결측값 처리
Gyeonggi_lamp2 = Gyeonggi_lamp2.fillna(0)
Gyeonggi_lamp2


# In[6]:


# 기타 컬럼의 ","을 제거한 후 float형으로 바꿈
Gyeonggi_lamp2['기타'] = Gyeonggi_lamp2['기타'].str.replace(",","")
Gyeonggi_lamp2['기타'] = Gyeonggi_lamp2['기타'].astype(float)
Gyeonggi_lamp2


# In[7]:


# 가로등의 총 합계
Gyeonggi_lamp_sum = Gyeonggi_lamp2.sum()
Gyeonggi_total = 0
for i in Gyeonggi_lamp_sum[1:] :
    Gyeonggi_total += int(i)
Gyeonggi_total


# ### 서울 가로등 개수

# In[8]:


seoul_lamp = pd.read_csv('./서울_가로등현황.csv', skiprows=1)

seoul_lamp2 = seoul_lamp.copy()
seoul_lamp2 = seoul_lamp2.iloc[1:, -3:-2]
seoul_lamp2['가로등'] = seoul_lamp2['가로등'].astype(int)
# seoul_lamp2['가로등']의 row수가 가로등 개수
seoul_total = seoul_lamp2['가로등'].sum()
seoul_total


# ### 부산 가로등 개수

# In[9]:


busan_lamp = pd.read_csv('./부산_가로등현황.csv', encoding='euc-kr')

busan_lamp2 = busan_lamp.copy()
busan_lamp2


# In[10]:


# 데이터의 전체 합계 수가 가로등 개수
busan_lamp_sum = busan_lamp2.sum()
busan_total = 0
for i in busan_lamp_sum[1:] :
    busan_total += int(i)
busan_total


# ### 대구 가로등 개수

# In[11]:


daegu_lamp = pd.read_csv('./대구_가로등현황.csv',encoding='euc-kr')

daegu_lamp2 = daegu_lamp.copy()
# 전체 row수가 가로등 개수
daegu_total = daegu_lamp2.shape[0]
daegu_total


# ### 인천 가로등 개수

# In[12]:


incheon_lamp = pd.read_csv('./인천_가로등현황.csv',encoding='euc-kr')

incheon_lamp2 = incheon_lamp.copy()
# 가로등 컬럼만 불러온다
incheon_lamp2 = incheon_lamp.iloc[:, -1]
incheon_total = incheon_lamp2.sum()
incheon_total


# ### 광주 가로등 개수

# In[13]:


# 광주 가로등 폴더에 있는 모든 csv 파일을 불러옴
gwangju_files = glob('./광주 가로등/*.csv')

gwangju_lamp = pd.DataFrame()
# 오류가 났을 때 파일 확인을 위한 tqdm 사용
for file_name in tqdm(gwangju_files):
    try :
        temp = pd.read_csv(file_name, encoding='euc-kr')
    except:
        temp = pd.read_csv(file_name)
    gwangju_lamp = pd.concat([gwangju_lamp, temp]) # 모든 csv 데이터를 합침
# 합친 csv 파일의 row값 = 가로등 개수
gwangju_total = gwangju_lamp.shape[0]
gwangju_total


# ### 대전 가로등 개수

# In[14]:


daejeon_lamp = pd.read_csv('./대전_가로등현황.csv', encoding='euc-kr')

daejeon_lamp2 = daejeon_lamp.copy()
# row 값 = 가로등 개수
daejeon_total = daejeon_lamp2.shape[0]
daejeon_total


# ### 울산 가로등 개수

# In[15]:


ulsan_lamp = pd.read_csv('./울산_가로등현황.csv', encoding='euc-kr')

ulsan_lamp2 = ulsan_lamp.copy()
ulsan_lamp2


# In[16]:


ulsan_lamp3 = ulsan_lamp2['지형지물부호'] == '가로등'
ulsan_total = 0
# 지형지물부호가 가로등인 row만 count
for i in ulsan_lamp3 :
    if i == True :
        ulsan_total += 1
ulsan_total


# ### 세종 가로등 개수

# In[17]:


sejong_lamp = pd.read_csv('./세종_가로등현황.csv', encoding='euc-kr')

sejong_lamp2 = sejong_lamp.copy()
# row값 = 가로등 개수
sejong_total = sejong_lamp2.shape[0]
sejong_total


# ### 강원, 제주 가로등 개수

# In[18]:


# 자료 x, 기사를 통해 정보 얻음
gangwon_total = 98482
jeju_total = 34683


# ### 경북 가로등 개수

# In[19]:


# 경북 가로등 폴더에 있는 모든 csv 파일을 불러옴
Gyeongsangbuk_files = glob('./경북 가로등/*.csv')

Gyeongsangbuk_lamp = pd.DataFrame()
# 오류가 났을 때 파일 확인을 위한 tqdm 사용
for file_name in tqdm(Gyeongsangbuk_files):
    try :
        temp = pd.read_csv(file_name, encoding='euc-kr')
    except :
        temp = pd.read_csv(file_name)
    Gyeongsangbuk_lamp = pd.concat([Gyeongsangbuk_lamp, temp]) # 모든 csv 파일을 합침
# 합친 csv 파일의 row수 = 가로등 개수
Gyeongsangbuk_total = Gyeongsangbuk_lamp.shape[0]
Gyeongsangbuk_total


# ### 경남 가로등 개수

# In[20]:


# 경남 가로등 폴더에 있는 모든 csv 파일을 불러옴
Gyeongsangnam_files = glob('./경남 가로등/*.csv')

Gyeongsangnam_lamp = pd.DataFrame()
# 오류가 났을 때 파일 확인을 위한 tqdm 사용
for file_name in tqdm(Gyeongsangnam_files):
    try :
        temp = pd.read_csv(file_name, encoding='euc-kr')
    except:
        temp = pd.read_csv(file_name)
    Gyeongsangnam_lamp = pd.concat([Gyeongsangnam_lamp, temp]) # 모든 csv 파일을 합침
# 합친 csv 파일의 row수 = 가로등 개수
Gyeongsangnam_total = Gyeongsangnam_lamp.shape[0]
Gyeongsangnam_total


# ### 충남 가로등 개수

# In[21]:


# 충남 가로등 폴더에 있는 모든 csv 파일을 불러옴
Chungcheongnam_files = glob('./충남 가로등/*.csv')

Chungcheongnam_lamp = pd.DataFrame()
# 오류가 났을 때 파일 확인을 위한 tqdm 사용
for file_name in tqdm(Chungcheongnam_files):
    try :
        temp = pd.read_csv(file_name, encoding='euc-kr')
    except:
        temp = pd.read_csv(file_name)
    Chungcheongnam_lamp = pd.concat([Chungcheongnam_lamp, temp]) # 모든 csv 파일을 합침
# 합친 csv 파일의 row수 = 가로등 개수
Chungcheongnam_total = Chungcheongnam_lamp.shape[0]
Chungcheongnam_total


# ### 충북 가로등 개수

# In[22]:


# 에러
# 해결 정민님 감사합니다!!
# 충북 가로등 폴더에 있는 모든 csv 파일을 불러옴
Chungcheongbuk_files = glob('./충북 가로등/*.csv')

Chungcheongbuk_lamp = pd.DataFrame()
# 오류가 났을 때 파일 확인을 위한 tqdm 사용
for file_name in tqdm(Chungcheongbuk_files):
    try :
        temp = pd.read_csv(file_name, encoding='euc-kr')
    except :
        try :
            temp = pd.read_csv(file_name)
        except :
            temp = pd.read_csv(file_name, encoding='cp949') # euc-kr, utf-8 인코딩 모두 오류시 cp949 인코딩
    Chungcheongbuk_lamp = pd.concat([Chungcheongbuk_lamp, temp]) # 모든 csv 파일을 합침
# 합친 csv 파일의 row수 = 가로등 개수
Chungcheongbuk_total = Chungcheongbuk_lamp.shape[0]
Chungcheongbuk_total


# ### 전북 가로등 개수

# In[23]:


# 전북 가로등 폴더에 있는 모든 csv 파일을 불러옴
Jeollabuk_files = glob('./전북 가로등/*.csv')

Jeollabuk_lamp = pd.DataFrame()
# 오류가 났을 때 파일 확인을 위한 tqdm 사용
for file_name in tqdm(Jeollabuk_files):
    try :
        temp = pd.read_csv(file_name, encoding='euc-kr')
    except:
        temp = pd.read_csv(file_name)
    Jeollabuk_lamp = pd.concat([Jeollabuk_lamp, temp]) # 모든 csv 파일을 합침
# 합친 csv 파일의 row수 = 가로등 개수
Jeollabuk_total = Jeollabuk_lamp.shape[0]
Jeollabuk_total


# ### 전남 가로등 개수

# In[24]:


# 전남 가로등 폴더에 있는 모든 csv 파일을 불러옴
Jeollanam_files = glob('./전남 가로등/*.csv')

Jeollanam_lamp = pd.DataFrame()
# 오류가 났을 때 파일 확인을 위한 tqdm 사용
for file_name in tqdm(Jeollanam_files):
    try :
        temp = pd.read_csv(file_name, encoding='euc-kr')
    except:
        temp = pd.read_csv(file_name)
    Jeollanam_lamp = pd.concat([Jeollanam_lamp, temp]) # 모든 csv 파일을 합침
# 합친 csv 파일의 row수 = 가로등 개수
Jeollanam_total = Jeollanam_lamp.shape[0]
Jeollanam_total


# In[25]:


# 전국 가로등 개수를 리스트에 추가
total_lamp = [seoul_total, busan_total, daegu_total, incheon_total, gwangju_total, daejeon_total, ulsan_total, sejong_total, 
              Gyeonggi_total, gangwon_total, Chungcheongbuk_total, Chungcheongnam_total, Jeollabuk_total, Jeollanam_total, 
              Gyeongsangbuk_total, Gyeongsangnam_total, jeju_total]
total_lamp


# ## 지역별 범죄 건수

# In[26]:


crime = pd.read_csv('./지역별 범죄 건수, 지역별 인구수.csv', encoding='euc_kr')

# 원본 훼손 방지를 위해 copy
crime2 = crime.copy()
crime2.columns = ['지역', '2017', '2018', '2019', '2020', '2021']
crime2 = crime2.set_index(["지역"])
crime2


# In[27]:


crime.info()


# In[28]:


# 결측치 확인
pd.isna(crime2)


# In[29]:


# 중앙값 열 추가
crime2.loc[:,'중앙값'] = crime2.median(axis=1)
crime2


# In[30]:


# 수의 차이
# 각 항목의 최소값을 0, 최대값을 1로 설정하고 범위 안에서 비교

col =  ['2017', '2018', '2019', '2020', '2021','중앙값']
x = crime2[col].values
scaler = preprocessing.MinMaxScaler()             # 정규화 시켜준다.
x_scale =scaler.fit_transform(x.astype(float))
crime_minmax = pd.DataFrame(x_scale, columns=col, index = crime2.index)
crime_minmax


# ## 월별 범죄 발생 건수

# In[31]:


monthCrime = pd.read_csv('./월별 범죄 발생 건수.csv', encoding='euc_kr')

# 원본 훼손 방지를 위해 copy
monthCrime2 = monthCrime.copy()

monthCrime2.columns = ['월', '2017', '2018', '2019', '2020', '2021']
monthCrime2


# In[32]:


# 칼럼값을 프레임 내에 넣기 위해 stack
monthCrime2 = pd.DataFrame(monthCrime2.stack()).reset_index()
monthCrime2


# In[33]:


monthCrime2 = monthCrime2.rename(columns={'level_0' : '월', 'level_1' : '연도', 0 : '총 범죄수'})
monthCrime2['월'] = monthCrime2['월']+1
monthCrime2


# In[34]:


# 잘못 들어간 행 삭제
for i in range(0, 71) :
    if monthCrime2.loc[i]['연도'] == '월' :
        monthCrime2 = monthCrime2.drop(i, axis = 0)
monthCrime2


# In[35]:


# 인덱스 재배열
monthCrime2 = monthCrime2.reset_index()
monthCrime2 = monthCrime2.iloc[:, 1:4]
monthCrime2


# ## 일별 범죄 발생 건수

# In[36]:


dayCrime = pd.read_csv('./요일별 범죄 발생 건수.csv', encoding='euc_kr')

# 원본 훼손 방지를 위해 copy
dayCrime2 = dayCrime.copy()
dayCrime2.columns = ['요일', '2017', '2018', '2019', '2020', '2021']
dayCrime2 = dayCrime2.set_index(["요일"])


# In[37]:


# 칼럼값을 프레임 내에 넣기 위해 stack
dayCrime2 = dayCrime2.stack().reset_index()
dayCrime2 = dayCrime2.rename(columns={'level_1' : '연도', 0 : '총 범죄수'})
dayCrime2


# In[38]:


dayCrime2.info()


# ## 시간별 범죄 발생 건수

# In[39]:


timeCrime = pd.read_csv('./시간별 범죄 발생 건수.csv', encoding='euc_kr')

# 원본 훼손 방지를 위해 copy
timeCrime2 = timeCrime.copy()
timeCrime2.columns = ['시간', '2017', '2018', '2019', '2020', '2021']
timeCrime2 = timeCrime2.set_index(["시간"])
timeCrime2

# timeCrime2 = timeCrime.copy()
# timeCrime2['year'] = timeCrime2['year'].astype(str)


# In[40]:


# 칼럼값을 프레임 내에 넣기 위해 stack
timeCrime2 = timeCrime2.stack().reset_index()
timeCrime2


# In[41]:


# 컬럼명 변경
timeCrime2 = timeCrime2.rename(columns={'level_1' : '연도', 0 : '총 범죄수'})
timeCrime2


# In[42]:


timeCrime2.info()


# ## 장소별 범죄 발생 건수

# In[43]:


placeCrime = pd.read_csv('./장소별 범죄 건수.csv', encoding='euc_kr')

# 원본 훼손 방지를 위해 copy
placeCrime2 = placeCrime.copy()
placeCrime2.columns = ['장소', '2017', '2018', '2019', '2020', '2021']
placeCrime3 = placeCrime2.copy()
placeCrime2 = placeCrime2.set_index(['장소'])

# 컬럼값(장소)을 데이터에 넣음
placeCrime2 = pd.DataFrame(placeCrime2.stack()).reset_index()

# 장소의 실외 여부 리스트
in_or_out = [0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0,
             0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 
             1, 1, 0, 0, 0]

placeCrime3.insert(6, '실외 여부',in_or_out)
placeCrime3 = placeCrime3.drop(columns='장소')
placeCrime3 = placeCrime3.set_index(['실외 여부'])
# 컬럼값(실외 여부)을 데이터에 넣음
placeCrime3 = pd.DataFrame(placeCrime3.stack()).reset_index()
# 실외 여부 데이터만 추출
placeCrime3 = placeCrime3.iloc[:, :1]
placeCrime2.insert(3, '실외 여부', placeCrime3)
placeCrime2 = placeCrime2.rename(columns={'level_1' : '연도', 0 : '총 범죄수'})
placeCrime2['실외 여부'] = placeCrime2['실외 여부'].astype(str)

# 불필요한 '기타' 행 제거
placeCrime2 = placeCrime2.iloc[:-5]
placeCrime2


# In[44]:


# # 불필요한 '기타' 행 제거
# placeCrime2 = placeCrime2.drop(['기타'],axis=0)
# placeCrime2
placeCrime2 = placeCrime2.rename(columns={'level_1' : '연도', 0 : '총 범죄수'})
# 불필요한 '기타' 행 제거
placeCrime2 = placeCrime2.iloc[:-5]
placeCrime2


# In[45]:


# # 합계 열 추가
# placeCrime2.loc[:, '합계'] = placeCrime2.loc[:,'2017':'2021'].sum(axis=1)
# placeCrime2


# In[46]:


placeCrime2.info()


# ## 경찰서,파출소,지구대 현황

# In[47]:


police = pd.read_csv('./1720전국 경찰서 현황.csv', encoding='euc_kr')

# 원본 훼손 방지를 위해 copy
police2 = police.copy()
police2


# In[48]:


# 결측치 갯수 세기
police2.isnull().sum(axis=1) # 행으로 계산


# In[49]:


# '세종'의 결측치를 2019년도 정보로 대체
police2['17년도 지구대'] = police['17년도 지구대'].fillna(0)
police2['17년도 파출소'] = police['17년도 파출소'].fillna(0)
police2['18년도 지구대'] = police['18년도 지구대'].fillna(0)
police2['18년도 파출소'] = police['18년도 파출소'].fillna(0)

police2


# In[50]:


# 결측치 갯수 재확인
police2.isnull().sum(axis=1) # 행으로 계산


# In[51]:


# 년도별로 열 합치기
for i in range(17,22):
    i = str(i)
    police2.loc[:, i+'년도 경찰서,지구대,파출소'] = police2.loc[:,i+'년도 경찰서':i+'년도 파출소'].sum(axis=1)

# 불필요한 열 제거
for i in range(17,22):
    i = str(i)
    police2 = police2.drop([i+'년도 경찰서',i+'년도 지구대',i+'년도 파출소'],axis=1)

police2.columns = ['지역', '17년도 경찰서,지구대,파출소', '18년도 경찰서,지구대,파출소', '19년도 경찰서,지구대,파출소',
                  '20년도 경찰서,지구대,파출소','21년도 경찰서,지구대,파출소']
police2 = police2.set_index(["지역"])
police2


# In[52]:


police2.loc['경기'] = police2.loc['경기남부':'경기북부'].sum()

# 행 순서 변경
police2 = police2.reindex(index=['서울','부산','대구','인천','광주','대전','울산','세종','경기','강원',
                        '충북','충남','전북','전남','경북','경남','제주'])
police2


# In[53]:


'''
# 합계 행 추가
police3 = police2
police3.loc['합계',:] = police3.loc['서울':'제주'].sum(axis=0)
police3
'''


# In[54]:


# 수의 차이
# 각 항목의 최소값을 0, 최대값을 1로 설정하고 범위 안에서 비교

col =  ['17년도 경찰서,지구대,파출소', '18년도 경찰서,지구대,파출소', '19년도 경찰서,지구대,파출소',
        '20년도 경찰서,지구대,파출소','21년도 경찰서,지구대,파출소']
x = police2[col].values
scaler = preprocessing.MinMaxScaler()             # 정규화 시켜준다.
x_scale =scaler.fit_transform(x.astype(float))
police_minmax = pd.DataFrame(x_scale, columns=col, index = police2.index)
police_minmax


# In[55]:


# 지역별 범죄 발생 건수와 합치기 (일반 데이터)
crime_df = crime2.merge(police2[['17년도 경찰서,지구대,파출소', '18년도 경찰서,지구대,파출소', '19년도 경찰서,지구대,파출소',
                                 '20년도 경찰서,지구대,파출소','21년도 경찰서,지구대,파출소']], on='지역')
crime_df


# In[56]:


# 지역별 범죄 발생 건수와 합치기 (정규화 데이터)
crime_df_minmax = crime_minmax.merge(police_minmax[['17년도 경찰서,지구대,파출소', '18년도 경찰서,지구대,파출소', '19년도 경찰서,지구대,파출소',
                                 '20년도 경찰서,지구대,파출소','21년도 경찰서,지구대,파출소']], on='지역')
crime_df_minmax


# ## CCTV 설치 운영 현황

# In[57]:


cctv = pd.read_csv('./CCTV_설치_운영_현황_20230913163452.csv', encoding='euc_kr')

# 원본 훼손 방지를 위해 copy
cctv2 = cctv.copy()
cctv2


# In[58]:


# 불필요한 columns 제거
cctv2 = cctv2.drop(['구분(1)','전체 사업체 (개)','CCTV 미설치/미운영 사업체 수 (개)', 'CCTV 미설치/미운영 사업체 비율 (%)'], axis=1)
# 불필요한 rows 제거
cctv2 = cctv2.drop([0, 18, 19, 20, 21, 22], axis=0)
cctv2


# In[59]:


cctv2.columns = ['지역', 'CCTV 설치/운영 사업체 수 (개)', 'CCTV 설치/운영 사업체 비율 (%)', 'CCTV 설치/운영 대수 (대)']
cctv2 = cctv2.set_index(["지역"])
cctv2


# In[60]:


# 결측치 확인
pd.isna(cctv2)


# In[61]:


# 수의 차이
# 각 항목의 최소값을 0, 최대값을 1로 설정하고 범위 안에서 비교

col =  ['CCTV 설치/운영 사업체 수 (개)', 'CCTV 설치/운영 사업체 비율 (%)', 'CCTV 설치/운영 대수 (대)']
x = cctv2[col].values
scaler = preprocessing.MinMaxScaler()             # 정규화 시켜준다.
x_scale =scaler.fit_transform(x.astype(float))
cctv_minmax = pd.DataFrame(x_scale, columns=col, index = cctv2.index)
cctv_minmax


# In[62]:


# 지역별 범죄 발생 건수와 합치기 (일반 데이터)
# CCTV 운영대수 열만 가져와서 합치기

crime_df = crime_df.merge(cctv2[['CCTV 설치/운영 대수 (대)']], on='지역')
crime_df


# In[63]:


# 지역별 범죄 발생 건수와 합치기 (정규화 데이터)
# CCTV 운영대수 열만 가져와서 합치기

crime_df_minmax = crime_df_minmax.merge(cctv_minmax[['CCTV 설치/운영 대수 (대)']], on='지역')
crime_df_minmax


# ## 외국인 비율

# In[64]:


foreigner = pd.read_csv('./외국인__시군구_20230913163658.csv', encoding='euc_kr')

# 원본 훼손 방지를 위해 copy
foreigner2 = foreigner.copy()
foreigner2


# In[65]:


foreigner2 = foreigner2.drop([0, 1, 2, 3, 4], axis=0)
foreigner2 = foreigner2.drop('행정구역별(시군구)',axis=1)
foreigner2


# In[66]:


# 열 이름 변경
foreigner2 = foreigner2.rename(columns={'2017':'2017_외국인','2018':'2018_외국인','2019':'2019_외국인','2020':'2020_외국인','2021':'2021_외국인'})

# 열 추가
foreigner2['지역'] = ['서울','부산','대구','인천','광주','대전','울산','세종','경기','강원',
                        '충북','충남','전북','전남','경북','경남','제주']

# 열 순서 변경
foreigner2 = foreigner2.reindex(columns=['지역','2017_외국인','2018_외국인', '2019_외국인', '2020_외국인', '2021_외국인'])

# index 지정
foreigner2 = foreigner2.set_index(['지역'])
foreigner2


# In[67]:


foreigner2.info()


# In[68]:


# 결측치 확인
pd.isna(foreigner2)


# In[69]:


# 수의 차이
# 각 항목의 최소값을 0, 최대값을 1로 설정하고 범위 안에서 비교

col =  ['2017_외국인','2018_외국인', '2019_외국인', '2020_외국인', '2021_외국인']
x = foreigner2[col].values
scaler = preprocessing.MinMaxScaler()             # 정규화 시켜준다.
x_scale =scaler.fit_transform(x.astype(float))
foreigner_minmax = pd.DataFrame(x_scale, columns=col, index = foreigner2.index)
foreigner_minmax


# In[70]:


# 지역별 범죄 발생 건수와 합치기 (일반 데이터)
crime_df = crime_df.merge(foreigner2[['2017_외국인','2018_외국인', '2019_외국인', '2020_외국인', '2021_외국인']], on='지역')
crime_df


# In[71]:


# 지역별 범죄 발생 건수와 합치기 (정규화 데이터)
crime_df_minmax = crime_df_minmax.merge(foreigner_minmax[['2017_외국인','2018_외국인', '2019_외국인', '2020_외국인', '2021_외국인']], on='지역')
crime_df_minmax


# ## 평균 연령

# In[72]:


age = pd.read_csv('./201701_202112_주민등록인구기타현황(평균연령)_avgAge.csv', encoding='euc_kr')

# 원본 훼손 방지를 위해 copy
age2 = age.copy()
age2


# In[73]:


# 불필요한 열 삭제(성별 평균 연령)

for i in range(2017,2022):
    i = str(i)
    for j in range(1,10):
        j=str(j)
        age2 = age2.drop([i+'년0'+j+'월_남자 평균연령', i+'년0'+j+'월_여자 평균연령'],axis=1)

    for k in range(10,13):
        k=str(k)
        age2 = age2.drop([i+'년'+k+'월_남자 평균연령', i+'년'+k+'월_여자 평균연령'],axis=1)
age2


# In[74]:


# 결측치 갯수 세기
age.isnull().sum(axis=1) # 행으로 계산


# In[75]:


# 월별로 나누어져 있는 데이터를 연도별로 합치기
    
for i in range(2017,2022):
    i = str(i)
    age2.loc[:, i+'년_평균연령'] = round(age2.loc[:,(i+'년01월_평균연령'):(i+'년12월_평균연령')].sum(axis=1)/12,2)
age2


# In[76]:


# 불필요한 열 삭제
for i in range(2017,2022):
    i = str(i)
    m = ['01','02','03','04','05','06','07','08','09','10','11','12']
    for j in m:
        age2 = age2.drop([i+'년'+j+'월_평균연령'],axis=1)

# 불필요한 행 삭제
age2 = age2.drop(0, axis=0)

age2


# In[77]:


age2.columns = ['지역', '2017년_평균연령', '2018년_평균연령', '2019년_평균연령', '2020년_평균연령', '2021년_평균연령']
age2 = age2.set_index(["지역"])
age2


# In[78]:


# 행 이름 변경
age2 = age2.rename(index={'서울특별시  (1100000000)':'서울','부산광역시  (2600000000)':'부산','대구광역시  (2700000000)':'대구',
                          '인천광역시  (2800000000)':'인천','광주광역시  (2900000000)':'광주','대전광역시  (3000000000)':'대전',
                         '울산광역시  (3100000000)':'울산','세종특별자치시  (3600000000)':'세종','경기도  (4100000000)':'경기',
                         '강원도  (4200000000)':'강원','충청북도  (4300000000)':'충북','충청남도  (4400000000)':'충남',
                         '전라북도  (4500000000)':'전북', '전라남도  (4600000000)':'전남','경상북도  (4700000000)':'경북',
                         '경상남도  (4800000000)':'경남','제주특별자치도  (5000000000)':'제주'})
age2


# In[79]:


# 수의 차이
# 각 항목의 최소값을 0, 최대값을 1로 설정하고 범위 안에서 비교

col =  ['2017년_평균연령', '2018년_평균연령', '2019년_평균연령', '2020년_평균연령', '2021년_평균연령']
x = age2[col].values
scaler = preprocessing.MinMaxScaler()             # 정규화 시켜준다.
x_scale =scaler.fit_transform(x.astype(float))
age_minmax = pd.DataFrame(x_scale, columns=col, index = age2.index)
age_minmax


# In[80]:


# 지역별 범죄 발생 건수와 합치기 (일반 데이터)
crime_df = crime_df.merge(age2[['2017년_평균연령', '2018년_평균연령', '2019년_평균연령', '2020년_평균연령', '2021년_평균연령']], on='지역')
crime_df


# In[81]:


# 지역별 범죄 발생 건수와 합치기 (정규화 데이터)
crime_df_minmax = crime_df_minmax.merge(age_minmax[['2017년_평균연령', '2018년_평균연령', '2019년_평균연령', '2020년_평균연령', '2021년_평균연령']], on='지역')
crime_df_minmax


# ## 인구수

# In[82]:


num = pd.read_csv('./201712_202112_주민등록인구및세대현황_연간.csv', encoding='euc_kr')

# 원본 훼손 방지를 위해 copy
num2 = num.copy()
num2


# In[83]:


# 불필요한 열 제거

num2 = num2.drop(['2017년_세대수','2017년_세대당 인구', '2017년_남자 인구수', '2017년_여자 인구수','2017년_남여 비율', 
                  '2018년_세대수','2018년_세대당 인구', '2018년_남자 인구수', '2018년_여자 인구수','2018년_남여 비율', 
                  '2019년_세대수','2019년_세대당 인구', '2019년_남자 인구수', '2019년_여자 인구수','2019년_남여 비율', 
                  '2020년_세대수','2020년_세대당 인구', '2020년_남자 인구수', '2020년_여자 인구수','2020년_남여 비율', 
                  '2021년_세대수','2021년_세대당 인구', '2021년_남자 인구수', '2021년_여자 인구수','2021년_남여 비율'],axis=1)

# 불필요한 행 제거
num2 = num2.drop([0])
num2


# In[84]:


# 열 이름 변경
# num2 = num2.rename(columns={'행정구역':'지역','2023년08월_총인구수':'총인구수'})
num2['2017년_총인구수'] = num2['2017년_총인구수'].str.replace(",","").astype(int)
num2['2018년_총인구수'] = num2['2018년_총인구수'].str.replace(",","").astype(int)
num2['2019년_총인구수'] = num2['2019년_총인구수'].str.replace(",","").astype(int)
num2['2020년_총인구수'] = num2['2020년_총인구수'].str.replace(",","").astype(int)
num2['2021년_총인구수'] = num2['2021년_총인구수'].str.replace(",","").astype(int)
num2['총인구수'] = num2.loc[:, '2017년_총인구수' : '2021년_총인구수'].mean(axis=1)

# 불필요한 행 제거
num2 = num2.drop(columns=['2017년_총인구수', '2018년_총인구수', '2019년_총인구수', '2020년_총인구수', '2021년_총인구수'])
num2

# index 지정
num2.columns = ['지역', '총인구수']
num2 = num2.set_index(["지역"])
num2


# In[85]:


# 행 이름 변경
num2 = num2.rename(index={'서울특별시  (1100000000)':'서울','부산광역시  (2600000000)':'부산','대구광역시  (2700000000)':'대구',
                          '인천광역시  (2800000000)':'인천','광주광역시  (2900000000)':'광주','대전광역시  (3000000000)':'대전',
                         '울산광역시  (3100000000)':'울산','세종특별자치시  (3600000000)':'세종','경기도  (4100000000)':'경기',
                         '강원도  (4200000000)':'강원','충청북도  (4300000000)':'충북','충청남도  (4400000000)':'충남',
                         '전라북도  (4500000000)':'전북', '전라남도  (4600000000)':'전남','경상북도  (4700000000)':'경북',
                         '경상남도  (4800000000)':'경남','제주특별자치도  (5000000000)':'제주'})
num2


# In[86]:


# 지역별 범죄 발생 건수와 합치기 (일반 데이터)
crime_df = crime_df.merge(num2[['총인구수']], on='지역')
crime_df


# In[87]:


# 지역별 범죄 발생 건수와 합치기 (정규화 데이터)
crime_df_minmax = crime_df_minmax.merge(num2[['총인구수']], on='지역')
crime_df_minmax


# In[88]:


# 열 순서 변경 (일반 데이터)
crime_df = crime_df.reindex(columns=['총인구수','중앙값','2017','2018','2019','2020','2021','17년도 경찰서,지구대,파출소','18년도 경찰서,지구대,파출소','19년도 경찰서,지구대,파출소','20년도 경찰서,지구대,파출소','21년도 경찰서,지구대,파출소','CCTV 설치/운영 대수 (대)','2017_외국인','2018_외국인', '2019_외국인', '2020_외국인', '2021_외국인','2017년_평균연령', '2018년_평균연령', '2019년_평균연령', '2020년_평균연령', '2021년_평균연령'])
crime_df


# In[89]:


# 결측치 갯수 확인
crime_df.isnull().sum(axis=1) # 행으로 계산


# In[90]:


# 열 순서 변경 (정규화 데이터)
crime_df_minmax = crime_df_minmax.reindex(columns=['총인구수','2017','2018','2019','2020','2021','17년도 경찰서,지구대,파출소','18년도 경찰서,지구대,파출소','19년도 경찰서,지구대,파출소','20년도 경찰서,지구대,파출소','21년도 경찰서,지구대,파출소','CCTV 설치/운영 대수 (대)','2017_외국인','2018_외국인', '2019_외국인', '2020_외국인', '2021_외국인','2017년_평균연령', '2018년_평균연령', '2019년_평균연령', '2020년_평균연령', '2021년_평균연령'])
crime_df_minmax


# In[91]:


# 결측치 갯수 확인
crime_df_minmax.isnull().sum(axis=1) # 행으로 계산


# In[92]:


# 지도 시각화를 위한 도시의 영어이름 칼럼 추가
name_eng = ['Seoul', 'Busan', 'Daegu', 'Incheon', 'Gwangju', 'Daejeon', 'Ulsan', 
            'Sejongsi', 'Gyeonggi-do', 'Gangwon-do', 'Chungcheongbuk-do', 'Chungcheongnam-do', 
            'Jeollabuk-do', 'Jeollanam-do', 'Gyeongsangbuk-do', 'Gyeongsangnam-do', 'Jeju-do']
crime_df.insert(0, '지역(영어)', name_eng)
crime_df


# ## 공원 개수

# In[93]:


park_df = pd.read_csv('./전국도시공원정보표준데이터.csv', encoding='euc-kr')

park_df2 = park_df.copy()
park_total = park_df2['제공기관명']
# 결측값 확인
park_total.isna().sum()

# 문자열로 형변환
park_loc = list(park_total)

# 앞에 3글자만 가져옴
for i in range(len(park_loc) - 1)  :
    park_loc[i] = str(park_loc[i])[:3]
len(park_loc)


# In[94]:


# 전체 변수 초기화
gyeonggi_park = 0
gangwon_park = 0
seoul_park = 0
busan_park = 0
daegu_park = 0
incheon_park = 0
dajeon_park = 0
ulsan_park = 0
sejong_park = 0
Chungcheongbuk_park = 0
Chungcheongnam_park = 0
Jeollabuk_park = 0
Jeollanam_park = 0
Gyeongsangbuk_park = 0
Gyeongsangnam_park = 0
jeju_park = 0
gwangju_park = 0
# count 되지 않는 값 확인
error_park = []

# 공원 변수에 count
for i in range(len(park_loc) - 1) :
    if park_loc[i] == '경기도' :
        gyeonggi_park += 1
    elif park_loc[i] == '강원도' or park_loc[i] == '강원특' :
        gangwon_park += 1
    elif park_loc[i] == '서울특' :
        seoul_park += 1
    elif park_loc[i] == '부산광' or park_loc[i] == '기장군' or park_loc[i] == '부산관' :
        busan_park += 1
    elif park_loc[i] == '대구광' :
        daegu_park += 1
    elif park_loc[i] == '인천광' or park_loc[i] == '인천시' :
        incheon_park += 1
    elif park_loc[i] == '대전광' :
        dajeon_park += 1
    elif park_loc[i] == '울산광' or park_loc[i] == '울산시' :
        ulsan_park += 1
    elif park_loc[i] == '세종특' :
        sejong_park += 1
    elif park_loc[i] == '충청북' :
        Chungcheongbuk_park += 1
    elif park_loc[i] == '충청남' :
        Chungcheongnam_park += 1
    elif park_loc[i] == '전라북' :
        Jeollabuk_park += 1
    elif park_loc[i] == '전라남' :
        Jeollanam_park += 1
    elif park_loc[i] == '경상북' :
        Gyeongsangbuk_park += 1
    elif park_loc[i] == '경상남' :
        Gyeongsangnam_park += 1
    elif park_loc[i] == '제주특' :
        jeju_park += 1
    elif park_loc[i] == '광주광' :
        gwangju_park += 1
    else :
        error_park.append(park_loc[i])
error_park


# In[95]:


# 가져온 지역 별 공원 수를 리스트에 저장
total_park = [seoul_park, busan_park, daegu_park, incheon_park, gwangju_park, dajeon_park, ulsan_park, 
              sejong_park, gyeonggi_park, gangwon_park, Chungcheongbuk_park, Chungcheongnam_park,
              Jeollabuk_park, Jeollanam_park, Gyeongsangbuk_park, Gyeongsangnam_park, jeju_park]
total_park


# In[96]:


crime_df.insert(24, '공원 수', total_park)


# In[97]:


crime_df


# In[98]:


crime_df.insert(24, '가로등 수', total_lamp)


# In[99]:


crime_df


# ### 전국 편의점 개수

# In[100]:


# # 시간상 전체 주석 처리(코드 시연할때만 사용)
# # 5분 가량 소요
# # xml url을 불러와서 주소값만 추출
# address = []
# serviceKey = 'DEXLBUE5-DEXL-DEXL-DEXL-DEXLBUE508'
# for page_no in tqdm(range(1, 892)) :
#     url = f'https://www.safemap.go.kr/openApiService/data/getConvenienceStoreData.do?serviceKey={serviceKey}&pageNo={page_no}&numOfRows=50&dataType=XML&Fclty_Cd=509010'
#     request = requests.get(url).text
#     source = BeautifulSoup(request, features="xml")
#     txt = source.find_all('ADRES')
#     for i in txt :
#         address.append(i.get_text)
# len(address)


# In[101]:


# # 만약의 상황을 대비해 xlsx 파일로 저장
# pd.DataFrame(address).to_excel('전국 편의점 주소.xlsx')


# In[102]:


address = pd.read_excel('./전국 편의점 주소.xlsx')

address = list(address[0])


# In[103]:


# 주소 데이터 중 앞 3글자만 가져옴
for i in range(len(address)) :
    address[i] = str(address[i])[45:48]
address


# In[104]:


# 전체 변수 초기화
gyeonggi_store = 0
gangwon_store = 0
seoul_store = 0
busan_store = 0
daegu_store = 0
incheon_store = 0
dajeon_store = 0
ulsan_store = 0
sejong_store = 0
Chungcheongbuk_store = 0
Chungcheongnam_store = 0
Jeollabuk_store = 0
Jeollanam_store = 0
Gyeongsangbuk_store = 0
Gyeongsangnam_store = 0
jeju_store = 0
gwangju_store = 0
# count 되지 않는 값 확인
error_store = []

for i in address :
    if i == '경기도' :
        gyeonggi_store += 1
    elif i == '강원도' :
        gangwon_store += 1
    elif i == '서울특' :
        seoul_store += 1
    elif i == '부산광' :
        busan_store += 1
    elif i == '대구광' :
        daegu_store += 1
    elif i == '인천광' :
        incheon_store += 1
    elif i == '대전광' :
        dajeon_store += 1
    elif i == '울산광' :
        ulsan_store += 1
    elif i == '세종특' :
        sejong_store += 1
    elif i == '충청북' :
        Chungcheongbuk_store += 1
    elif i == '충청남' :
        Chungcheongnam_store += 1
    elif i == '전라북' :
        Jeollabuk_store += 1
    elif i == '전라남' :
        Jeollanam_store += 1
    elif i == '경상북' :
        Gyeongsangbuk_store += 1
    elif i == '경상남' :
        Gyeongsangnam_store += 1
    elif i == '제주특' :
        jeju_store += 1
    elif i == '광주광' :
        gwangju_store += 1
    else :
        error_store.append(i)
error_store


# In[105]:


total_store = [seoul_store, busan_store, daegu_store, incheon_store, gwangju_store, dajeon_store, ulsan_store, 
              sejong_store, gyeonggi_store, gangwon_store, Chungcheongbuk_store, Chungcheongnam_store,
              Jeollabuk_store, Jeollanam_store, Gyeongsangbuk_store, Gyeongsangnam_store, jeju_store]
total_store


# In[106]:


crime_df.insert(25, '편의점 수', total_store)


# In[107]:


crime_df


# ## 전국 코인노래방 개수

# In[108]:


karaoke_df = pd.read_excel('./코인,동전노래방_시도별.xlsx')

# 원본 손상 방지를 위해 copy
karaoke_df2 = karaoke_df.copy()
karaoke_df2


# In[109]:


karaoke_df2 = karaoke_df2.set_index(['지역'])
# 행 순서 변경
karaoke_df2 = karaoke_df2.reindex(['서울', '부산', '대구', '인천', '광주', '대전', 
                                   '울산', '세종', '경기', '강원', '충북', '충남', '전북', 
                                   '전남', '경북', '경남', '제주'])
karaoke_df2


# In[110]:


crime_df.insert(26, '코인노래방 수', karaoke_df2['개수'])
crime_df


# In[111]:


crime_df['2017_외국인']=crime_df['2017_외국인'].astype('float64')
crime_df['2018_외국인']=crime_df['2017_외국인'].astype('float64')
crime_df['2019_외국인']=crime_df['2017_외국인'].astype('float64')
crime_df['2020_외국인']=crime_df['2017_외국인'].astype('float64')
crime_df['2021_외국인']=crime_df['2017_외국인'].astype('float64')

crime_df.info()


# In[112]:


# 상관분석을 위해 데이터프레임 복사
crime_df1 = crime_df.copy()
# 2017~2021 까지의 총 범죄수 및 여러 요인의 평균 값 계산
crime_df1['총 범죄 수'] = crime_df1.loc[:,'2017':'2021'].sum(axis=1)
crime_df1['평균 치안센터 수'] = crime_df1.loc[:,'17년도 경찰서,지구대,파출소':'21년도 경찰서,지구대,파출소'].mean(axis=1)
crime_df1['평균 외국인 수'] = crime_df1.loc[:,'2017_외국인':'2021_외국인'].mean(axis=1)
crime_df1['평균연령'] = crime_df1.loc[:,'2017년_평균연령':'2021년_평균연령'].mean(axis=1)
crime_df1.info()


# In[113]:


# 인구 수 비례 여러 요인들의 상대적 크기 비교
crime_df1.insert(0, '인구 100명당 cctv 설치/운영 대수(대)',(crime_df1['CCTV 설치/운영 대수 (대)'] / crime_df1['총인구수'] * 100))
crime_df1.insert(0, '인구 100명당 총 범죄 수',(crime_df1['총 범죄 수'] / crime_df1['총인구수'] * 100))
crime_df1.insert(0, '인구 100명당 평균 치안센터 수',(crime_df1['평균 치안센터 수'] / crime_df1['총인구수'] * 100))
crime_df1.insert(0, '인구 100명당 평균 외국인 수',(crime_df1['평균 외국인 수'] / crime_df1['총인구수'] * 100))
crime_df1.insert(0, '인구 100명당 가로등 수',(crime_df1['가로등 수'] / crime_df1['총인구수'] * 100))
crime_df1.insert(0, '인구 100명당 공원 수',(crime_df1['공원 수'] / crime_df1['총인구수'] * 100))
crime_df1.insert(0, '인구 100명당 편의점 수',(crime_df1['편의점 수'] / crime_df1['총인구수'] * 100))
crime_df1.insert(0, '인구 100명당 코인노래방 수',(crime_df1['코인노래방 수'] / crime_df1['총인구수'] * 100))
crime_df1 = crime_df1.drop('총인구수', axis=1)
crime_df1


# In[114]:


crime_df1 = crime_df1.drop(columns= crime_df1.iloc[:, 9:-1])
crime_df1


# In[115]:


crime_df1 = crime_df1.drop(columns={'지역(영어)'})
crime_df1


# In[116]:


# 칼럼 순서 변경
crime_df1 = crime_df1[['인구 100명당 총 범죄 수', '인구 100명당 cctv 설치/운영 대수(대)', '인구 100명당 공원 수',
                       '인구 100명당 가로등 수', '인구 100명당 평균 외국인 수', '인구 100명당 평균 치안센터 수',
                       '인구 100명당 편의점 수', '인구 100명당 코인노래방 수', '평균연령']]
crime_df1


# # 월별 범죄 발생 건수

# In[117]:


fig = px.bar(monthCrime2, x="월", y='총 범죄수', color='연도')
fig.show()


# # <span style="background-color:#fff5b1"><span style="font-famaiy:ELAND_Choice_M; font-size:14px;">모든 연도에 대하여 3월 ~ 7월 까지는 범죄율이 증가하고 8월 ~ 2월까지는 범죄율이 감소하는 경향을 보인다.</span></span>

# # 요일별 범죄 발생 건수

# In[118]:


fig = px.bar(dayCrime2, x="요일", y='총 범죄수', color='연도')
fig.show()


# # <span style="background-color:#fff5b1"><span style="font-famaiy:ELAND_Choice_M; font-size:14px;">일 ~ 목요일에 비해 금 ~ 토요일에 범죄가 많이 일어난다는 것을 확인할 수 있다.</span></span>

# # 시간별 범죄 발생 건수

# In[119]:


fig = px.bar(timeCrime2, x="시간", y='총 범죄수', color='연도')
fig.show()


# # <span style="background-color:#fff5b1"><span style="font-famaiy:ELAND_Choice_M; font-size:14px;">18:00~23:59 까지 범죄가 많이 일어나고 있으며 정각 이후에는 대폭 감소하는 것을 확인할 수 있다.</span></span>

# # 장소별 범죄 발생 건수

# In[120]:


fig = px.bar(placeCrime2, x="장소", y='총 범죄수', color = '실외 여부', animation_frame='연도')
fig.show()


# # <span style="background-color:#fff5b1"><span style="font-famaiy:ELAND_Choice_M; font-size:14px;">실내에서는 아파트, 연립, 다세대, 단독주택 등 주거시설이나 상점, 유흥접객업소, 숙박업소 등에서 범죄가 가장 많이 일어나고, 실외에서는 노상에서 가장 많이 범죄가 일어나는 것을 확인할 수 있다.</span></span>

# # 지역별 범죄 발생 건수 지도 시각화

# In[121]:


geo_json = 'https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2018/json/skorea-provinces-2018-geo.json'

# 지도 초기화
crime_map = folium.Map(
    location=[36.5, 127.5],
    tiles='OpenStreetMap',
    zoom_start=7)

# 맵에 데이터를 추가하는 함수 생성
def create_choropleth(year):
    choropleth = folium.Choropleth(
        geo_data=geo_json,
        name=f'{year}년',
        data=crime_df,
        columns=['지역(영어)', year],
        key_on='feature.properties.name_eng',
        fill_color='PuRd',
        fill_opacity=0.7,
        line_opacity=0.5,
        legend_name=f'{year}년 지역별 범죄 수',
        show=False  # 처음은 안보이게 설정
    )
    return choropleth

# choropleth 래이어 추가
years = ['2017', '2018', '2019', '2020', '2021']
choropleth_layers = {year: create_choropleth(year) for year in years}

# choropleth_layers에 저장된 choropleth을 맵에 붙임
for year, choropleth in choropleth_layers.items():
    choropleth.add_to(crime_map)

# layer_control 생성
folium.LayerControl(
    collapsed=False,  # Layer Control 숨기기
    autoZIndex=False,  # Don't assign automatic z-index
).add_to(crime_map)  # 맵에 레이어 컨트롤 붙임


crime_map


# In[122]:


# 상관계수 확인
crime_df1.corr()


# In[123]:


# 히트맵 그리기
plt.figure(figsize=(10, 10))
plt.title("여러 요인들과 범죄 발생률 간의 상관관계", y = 1.05, size = 15)
sns.heatmap(crime_df1.corr(), linewidths = 0.1, vmax = 1.0,
           square = True, cmap = 'Blues', linecolor = "white", annot = True, annot_kws = {"size" : 16})


# # 상관계수 검정

# In[124]:


import scipy.stats as stats
crime_y = crime_df1['인구 100명당 총 범죄 수'].values
for factor in ['인구 100명당 cctv 설치/운영 대수(대)', '인구 100명당 공원 수', 
               '인구 100명당 공원 수', '인구 100명당 가로등 수', '인구 100명당 평균 외국인 수',
               '인구 100명당 평균 치안센터 수', '인구 100명당 편의점 수', '인구 100명당 코인노래방 수', '평균연령'] :
    print(factor)
    crime_x = crime_df1[factor].values
    print('상관 계수 : {:.2f}'.format(stats.pearsonr(crime_x,crime_y)[0]))
    print('유의 확률 : {:.4f}'.format(stats.pearsonr(crime_x,crime_y)[1]))
    print('\n')
    


# # <span style="background-color:#fff5b1"><span style="font-famaiy:ELAND_Choice_M; font-size:14px;">각 변수의 상관계수의 유의확률은 다음과 같고 그 중에서 유의 수준 0.05보다 작은 '인구 100명당 cctv 설치/운영 대수(대)', '인구 100명당 편의점 수', '평균연령' 변수들만 상관계수가 가장 유의미하다고 생각된다.</span></span>

# ## <span style="background-color:#fff5b1"><span style="font-famaiy:ELAND_Choice_M; font-size:14px;">인구 100명당 총 범죄 수와 cctv 설치 / 운영 대수 간에 상관 계수는 -0.63로 강한 음의 상관관계를 보이는 것을 확인할 수 있다.</span></span>
# ## <span style="background-color:#fff5b1"><span style="font-famaiy:ELAND_Choice_M; font-size:14px;">지역 별 평균 연령과 인구 100명당 총 범죄 수간의 상관 계수는 0.51로 강한 양의 상관관계를 보이는 것을 확인할 수 있다.</span></span>
# ## <span style="background-color:#fff5b1"><span style="font-famaiy:ELAND_Choice_M; font-size:14px;">인구 100명당 편의점 수와 인구 100명당 총 범죄 수 간에 상관 계수는 0.52로 강한 양의 상관관계를 보이는 것을 확인할 수 있다.</span></span>

# In[125]:


# 총 범죄 수와 cctv 설치 / 운영 대수(대)의 산점도 그래프
fig = px.scatter(data_frame = crime_df1, x = '인구 100명당 cctv 설치/운영 대수(대)', y = '인구 100명당 총 범죄 수',
                 trendline = 'ols', trendline_color_override = 'black')
fig.show()


# In[126]:


# 총 범죄 수와 평균 연령의 산점도 그래프
fig = px.scatter(data_frame = crime_df1, x = '평균연령', y = '인구 100명당 총 범죄 수',
                 trendline = 'ols', trendline_color_override = 'black')
fig.show()


# In[127]:


# 총 범죄 수와 인구 100명당 편의점 수의 산점도 그래프
fig = px.scatter(data_frame = crime_df1, x = '인구 100명당 편의점 수', y = '인구 100명당 총 범죄 수',
                 trendline = 'ols', trendline_color_override = 'black')
fig.show()


# In[128]:


fig = px.scatter(data_frame = crime_df1, x = '인구 100명당 공원 수', y = '인구 100명당 총 범죄 수',
                 trendline = 'ols', trendline_color_override = 'black')
fig.show()


# # 회귀분석

# ## 회귀분석 전 데이터 스케일링

# In[129]:


crime_df1_index = list(crime_df1.index)
crime_df1_columns = list(crime_df1.columns)

scaler = StandardScaler()
scaler.fit(crime_df1)
crime_stand = scaler.transform(crime_df1)
crime_stand = pd.DataFrame(crime_stand, index=crime_df1_index, columns=crime_df1_columns)
crime_stand


# In[130]:


# 표준화된 데이터의 상관계수 확인
crime_stand.corr()


# In[131]:


stY = crime_stand['인구 100명당 총 범죄 수'].values
stY


# In[132]:


stX = crime_stand.loc[:, ['인구 100명당 cctv 설치/운영 대수(대)', '인구 100명당 편의점 수', '평균연령']].values
stX


# In[133]:


#Standardized 된 변수들로 선형회귀모형 적합
import statsmodels.api as sm;
results = sm.OLS(stY, sm.add_constant(stX)).fit()
results.summary()


# 검정통계량 F값은 5.472이고 회귀모형의 유의확률 p값은 0.0118에서 유의수준 0.05보다 작기때문에 귀무가설 "회귀모형은 유의하지 않다"를 기각한다. 따라서 회귀모형은 유의하며 수정된 결정계수는 0.456이다.

# $\widehat{총 범죄수} = 1.027e-15 -0.3853 \times cctv 설치/운영 대수 + 0.2294	\times 평균연령 + 0.3797 \times 편의점 수 $

# 각 독립변수의 유의확률은 유의수준인 0.05보다 크기 때문에 
# 각각의 요인이 종속변수인 총 범죄 수의 인과관계가 있다고 판단하기에는 어렵다.

# ## 다중공선성 평가

# In[134]:


thr_comp=crime_stand[['인구 100명당 cctv 설치/운영 대수(대)', '인구 100명당 편의점 수', '평균연령']]
thr_comp


# In[135]:


pd.DataFrame()
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif=pd.DataFrame()
vif["VIF Factor"]=[variance_inflation_factor(
    thr_comp.values,i) for i in range(thr_comp.shape[1])]
vif["features"] = thr_comp.columns
vif


# In[136]:


# Standardized 데이터에서, 평균연령과 cctv 대수의 산점도 그래프
fig = px.scatter(data_frame = crime_stand, 
                 x = '평균연령',
                 y = '인구 100명당 cctv 설치/운영 대수(대)',
                 trendline = 'ols', trendline_color_override = 'black')
fig.show()


# In[137]:


# Standardized 데이터에서, 평균연령과 편의점수의 산점도 그래프
fig = px.scatter(data_frame = crime_stand, 
                 x = '평균연령',
                 y = '인구 100명당 편의점 수',
                 trendline = 'ols', trendline_color_override = 'black')
fig.show()


# In[138]:


# Standardized 데이터에서, 평균연령과 편의점수의 산점도 그래프
fig = px.scatter(data_frame = crime_stand, 
                 x = '인구 100명당 cctv 설치/운영 대수(대)',
                 y = '인구 100명당 편의점 수',
                 trendline = 'ols', trendline_color_override = 'black')
fig.show()


# # 시계열 분석

# In[139]:


# 원본 훼손 방지를 위해 copy
monthCrime3 = monthCrime2.copy()
monthCrime3


# In[140]:


# 원본데이터 사용 시
monthCrime3.info()


# In[141]:


# 원본 데이터 사용 시
monthCrime3['datetime'] = pd.to_datetime(monthCrime3['연도'] + '-' + monthCrime3['월'].astype(str) + '-1')
monthCrime3 = monthCrime3.rename(columns={'총 범죄수' : 'total'})
monthCrime3.info()


# In[142]:


# 시간 순으로 정렬
monthCrime3 = monthCrime3.sort_values(by = 'datetime')
monthCrime3


# In[143]:


# 총 범죄수와 datetime 컬럼만 추출해서 datetime을 인덱스로 지정
monthCrime3 = monthCrime3.iloc[:, 2:4].set_index('datetime')
monthCrime3.info()


# In[144]:


# 시간 순으로 lineplot 출력
sns.lineplot(data=monthCrime3, x = 'datetime', y = 'total')


# In[145]:


# 시계열 모델 생성
model_series = tsa.seasonal_decompose(monthCrime3['total'], model = 'additive')

# 모델 시각화
fig = model_series.plot()
plt.show()


# <span style="background-color:#fff5b1"><span style="font-famaiy:ELAND_Choice_M; font-size:14px;"> Trend는 감소하는 경향을 보이고, 계절성을 띄고 있음을 확인할 수 있다.</span></span>

# In[146]:


y1 = monthCrime3['total']


# In[147]:


# p, d, q값 → 0~1 사이의 값
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


# In[148]:


# 결과값을 저장할 리스트 생성
param_list = []
param_seasonal_list = []
result_AIC_list = []


# In[149]:


# 모델 시뮬레이션
# 가장 적합한 모델 찾기
for param in pdq :
    for param_seasonal in seasonal_pdq :
        try :
            mod = tsa.statespace.SARIMAX(y1, order = param, 
                                         seasonal_order = param_seasonal, 
                                         enforce_stationarity = False, 
                                         enforce_invertibility = False)
            results = mod.fit()
            param_list.append(param)
            param_seasonal_list.append(param_seasonal)
            result_AIC_list.append(results.aic)
        except :
            continue


# In[150]:


print(len(result_AIC_list), len(param_list), len(param_seasonal_list))


# In[151]:


# AIC가 제일 낮은 모델 선택
ARIMA_list = pd.DataFrame({'Parameter' : param_list, 'Seasonal' : param_seasonal_list, 'AIC' : result_AIC_list})
ARIMA_list.sort_values(by = 'AIC')


# In[152]:


# 선택된 모델로 시계열 분석
mod = sm.tsa.statespace.SARIMAX(y1,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])


# <span style="background-color:#fff5b1">ar.L1, ma.L1, ar.S.L12에 대해서 각각의 p-value가 유의수준 0.05보다 크므로 귀무가설을 채택한다. 따라서 각각의 회귀계수는 유의미하지 않다.(상관성이 없다.)</span>

# In[153]:


results.plot_diagnostics(figsize=(16, 8))
plt.show()


# <span style="font-famaiy:ELAND_Choice_M; font-size:14px;">첫 번째 그래프는 잔차의 시계열 데이터로 잔차의 시계열이 평균 0을 중심으로 변동하는 것으로 보인다. 두 번째 그래프는 잔차의 히스토그램과 잔차의 정규분포를 그린 것으로 어느정도 잔차가 정규 분포를 따른다는 것을 확인할 수 있다. 세 번째 그래프는 대부분 빨간선 위에 그려지는 것을 확인할 수 있으며 이는 잔차가 정규성을 만족한다고 볼 수 있다. 네 번째 그래프는 자기상관함수로 0(자기 자신과의 관계)와 2를 제외한 모든 값이 임계치 안에 들어와 있는 것을 확인할 수 있으며 데이터셋 자체는 자기상관성이 없음을 확인할 수 있다.</span>

# <span style="background-color:#fff5b1">따라서 잔차는 백색잡음(정상성)이고, 정규성은 어느정도 만족한다.</span>

# In[154]:


# 검증 그래프 
pred = results.get_prediction(start=pd.to_datetime('2021'), dynamic=False)

pred_ci = pred.conf_int()

ax = y1.plot(label='총 범죄 수', figsize=(20, 10))
pred.predicted_mean.plot(ax=ax, label='예측 모델')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='gray', alpha=0.25)
ax.set_xlabel('날짜')
ax.set_ylabel('총 범죄 수')
plt.legend()
plt.show()


# In[155]:


pred_ci_prediction = pred_ci.copy()
pred_ci_prediction


# In[156]:


# 실제 데이터와 비교
pred_ci_prediction.insert(1, 'actual value', monthCrime3['2021':])
pred_ci_prediction


# In[157]:


# 24개월 간의 예측 모델 생성
pred_uc = results.get_forecast(steps=24)

pred_ci = pred_uc.conf_int()

ax = y1.plot(label='총 범죄 수', figsize=(20, 10))
pred_uc.predicted_mean.plot(ax=ax, label='예측 모델')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='gray', alpha=0.25)
ax.set_xlabel('날짜')
ax.set_ylabel('총 범죄 수')
plt.legend()
plt.show()


# In[158]:


pred_ci_forecast = pred_ci.copy()
pred_ci_forecast


# In[159]:


# 예측 평가를 위한 데이터프레임 생성
# 2022~2023년 6월까지의 월별 범죄건수 X
# 분기별 전체 범죄건수 데이터 수집
monthCrime_test_data = [344622, 398320, 412979, 414708, 377482]
monthCrime_test_df = pd.DataFrame({'actual value' : monthCrime_test_data}, index = ['2022년 1분기', '2022년 2분기', '2022년 3분기', 
                                                                 '2022년 4분기', '2023년 1분기'])
monthCrime_test_df


# In[160]:


# 평가를 위해 분기별 데이터로 변환
monthCrime_forecast_lowerdata = [sum(pred_ci.iloc[0:3, 0]), sum(pred_ci.iloc[3:6, 0]), sum(pred_ci.iloc[6:9, 0]), 
                                 sum(pred_ci.iloc[9:12, 0]), sum(pred_ci.iloc[12:15, 0])]
monthCrime_forecast_upperdata = [sum(pred_ci.iloc[0:3, 1]), sum(pred_ci.iloc[3:6, 1]), sum(pred_ci.iloc[6:9, 1]), 
                                 sum(pred_ci.iloc[9:12, 1]), sum(pred_ci.iloc[12:15, 1])]

# 비교를 위해 insert
monthCrime_test_df.insert(0, 'lower value', monthCrime_forecast_lowerdata)
monthCrime_test_df.insert(2, 'upper value', monthCrime_forecast_upperdata)
monthCrime_test_df


# # <span style="background-color:#fff5b1"><span style="font-famaiy:ELAND_Choice_M; font-size:14px;">예측 모델에 비해 2022년 3분기 ~ 2023년 1분기 까지의 총 범죄수가 상당히 증가한 것을 확인할 수 있다.</span></span>

# # 예측 모델링
# 
# https://blog.naver.com/parktaij/223093724047
# xgboost, lightgbm 설치

# In[161]:


crime_df1.info()


# In[162]:


crime_df1.describe()     # dataframe의 통계적 요약


# # train-test-split
# 
# - crime_df1 데이터를 training 시키기 전에 train data와 test data로 나누어준다.
# - test 데이터의 갯수에 맞게 잘라서 X_train과 X_test에 각각 저장해 준다.
# - 타겟 변수인 '인구 100명당 총 범죄 수'를 따로 빼서 y에 저장해 준다.

# In[163]:


crime_df1


# In[164]:


from sklearn.model_selection import train_test_split

X_train = crime_df1[:8]
X_test=crime_df1[8:]

y_train = X_train['인구 100명당 총 범죄 수']
y_test = X_test['인구 100명당 총 범죄 수']

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape,  y_test.shape)


# # 데이터 스케일링
# 
# - 데이터가 가진 크기와 편차가 다르므로 데이터의 범위를 이상적으로 변환해 준다.
# - 데이터가 서로 비교될 수 있도록 StandardScaler를 통해 평균이 0이고 분산이 1인 정규 분포로 만들어 준다.

# In[165]:


scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)

X_val=scaler.transform(X_val)
X_test=scaler.transform(X_test)


# # Training
# 
# - 모델은 KNN , RandomForest,  XGB , Light GBM 활용
# - Grid Search를 사용하여 가장 우수한 성능을 보이는 모델의 하이퍼 파라미터 찾기

# In[166]:


# 모델은 KNN , RandomForest,  XGB , Light GBM 활용
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
import xgboost as xgb
from lightgbm.sklearn import LGBMRegressor

from sklearn.metrics import mean_squared_error


# In[167]:


knn = KNeighborsRegressor()
rf = RandomForestRegressor()
xgb = XGBRegressor()
lgb = LGBMRegressor()


# In[168]:


# fit -> 학습시킨다.
knn.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)
lgb.fit(X_train, y_train)

# predict -> 예측시킨다.
pred_crime = knn.predict(X_test)
pred_crime2 = rf.predict(X_test)
pred_crime3 = xgb.predict(X_test)
pred_crime4 = lgb.predict(X_test)

train_crime = knn.predict(X_train)
train_crime2 = rf.predict(X_train)
train_crime3 = xgb.predict(X_train)
train_crime4 = lgb.predict(X_train)

pred_val = knn.predict(X_val)
pred_val2 = rf.predict(X_val)
pred_val3 = xgb.predict(X_val)
pred_val4 = lgb.predict(X_val)

pred_crime = pred_crime.reshape(-1,1)
pred_crime2 = pred_crime2.reshape(-1,1)
pred_crime3 = pred_crime3.reshape(-1,1)
pred_crime4 = pred_crime4.reshape(-1,1)

mse_train = mean_squared_error(y_train, train_crime)
mse_val = mean_squared_error(y_val, pred_val)
mse_train2 = mean_squared_error(y_train, train_crime2)
mse_val2 = mean_squared_error(y_val, pred_val2)
mse_train3 = mean_squared_error(y_train, train_crime3)
mse_val3 = mean_squared_error(y_val, pred_val3)
mse_train4 = mean_squared_error(y_train, train_crime4)
mse_val4 = mean_squared_error(y_val, pred_val4)

print("1. KNN \t\t\t, train=%.4f, val=%.4f" % (mse_train, mse_val))
print("2. RF \t\t\t, train=%.4f, val=%.4f" % (mse_train2, mse_val2))
print("3. XGBoost \t\t\t, train=%.4f, val=%.4f" % (mse_train3, mse_val3))
print("4. LightGBM \t\t, train=%.4f, val=%.4f" % (mse_train4, mse_val4))

y_test1=y_test.values


# In[169]:


# Hyper-parameter tuning
# GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = {
    "max_depth" : [3,5,-1],         # 3
    "learning_rate" : [0.1, 0.01],  # 2
    "n_estimators" : [50, 100]      # 2
}

# 가능한 모든 조합들을 다 학습시킨다.
gscv = GridSearchCV(lgb, param_grid, scoring='neg_mean_squared_error', verbose=2)
gscv.fit(X_train, y_train)


# In[170]:


print(gscv.best_estimator_)


# In[171]:


print(gscv.best_params_)


# # Test(Prediction)
# 
# - 결과 분석 - 정확도 확인

# In[172]:


# training set과 같은 전처리
final_model = gscv.best_estimator_


# In[173]:


mse = mean_squared_error(y_test1, pred_crime)
mse2 = mean_squared_error(y_test1, pred_crime2)
mse3 = mean_squared_error(y_test1, pred_crime3)
mse4 = mean_squared_error(y_test1, pred_crime4)

print(mse)
print(mse2)
print(mse3)
print(mse4)


# In[174]:


print(np.sqrt(mse))
print(np.sqrt(mse2))
print(np.sqrt(mse3))
print(np.sqrt(mse4))


# # Summary

# In[175]:


# 평균 제곱 오차
print("-------------------- KNN --------------------")
print('MSE in training: %.7f' % mean_squared_error(y_test1, pred_crime))
print("-------------------- RF --------------------")
print('MSE in training:  %.7f' % mean_squared_error(y_test1, pred_crime2))
print("------------------ XGBoost ------------------")
print('MSE in training:  %.7f' % mean_squared_error(y_test1, pred_crime3))
print("------------------ LightGBM ------------------")
print('MSE in training:  %.7f' % mean_squared_error(y_test1, pred_crime4))

result_final = final_model.predict(X_test)
print("---------- Best LightGBM ----------")
print('MSE in training: %.7f' % mean_squared_error(y_test1, result_final))


# In[176]:


# y_test와 result_final을 사용하여 실제값과 예측값, 오차 출력

pred_r = pd.DataFrame(y_test)
pred_r['예측'] = result_final
pred_r['오차'] = (pred_r['인구 100명당 총 범죄 수']-pred_r['예측'])
pred_r


# In[ ]:




