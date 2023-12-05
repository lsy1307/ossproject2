import pandas as pd
from pandas import Series, DataFrame

data = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

print("1번\n")
con1 = ['H', 'avg', 'HR', 'OBP']
year1 = [2015, 2016, 2017, 2018]

for year in year1:
    print(str(year)+"년")
    for con in con1:
        data_top10 = data[data['year'].isin([year])]
        data_sorted = data_top10.sort_values(by=con, ascending=False)
        print(con+" :", data_sorted.head(10)['batter_name'].values)
        print("")

print("\n2번\n")
data_war = data[data['year'].isin([2018])]
pos1 = ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']

for pos in pos1:
    data_pos = data_war[data_war['cp'].isin([pos])]
    data_sorted = data_pos.sort_values(by='war', ascending=False)
    print(pos+" :", data_sorted.head(1)['batter_name'].values)
    
print("\n3번\n")
con2 = ['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG']
data_corr = data[con2].corrwith(data['salary']).sort_values(ascending=False)
print("Answer :", data_corr.head(1).index[0])
