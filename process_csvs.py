import os
import pandas as pd

my_dir = os.getcwd() + '/data_csvs/'
df_csvs = pd.concat(map(pd.read_csv,
                        [my_dir + 'fixtures_16-17.csv',
                         my_dir + 'fixtures_17-18.csv',
                         my_dir + 'fixtures_18-19.csv',
                         my_dir + 'fixtures_19-20.csv']))
pd.set_option('display.max_columns', None)  # enable printing of ALL columns

df_csvs['DATE'] = pd.to_datetime(df_csvs['DATE'], format="%d/%m/%Y")

df_csvs['TIME_'] = df_csvs.iloc[:, [2]]

df_csvs['time_test'] = df_csvs['TIME_'].str.extract(r'(\d+):\d+')

df_csvs['time_test'] = df_csvs['time_test'].astype(str).astype(int)

df_csvs.loc[df_csvs.time_test > 18, 'VENUE'] = 'NAC'
df_csvs.loc[df_csvs.time_test < 18, 'VENUE'] = 'TCD'

df_csvs['TIME__'] = df_csvs['TIME_'] + ':00'  # finish time format "hh:mm:ss"

df_csvs.drop('TIME_', inplace=True, axis=1)
df_csvs = df_csvs[['VENUE', 'DATE', 'TIME__', 'COMPETITION', 'HOME', 'AWAY', 'SCORE H', 'SCORE A']]
df_csvs.rename(columns={'COMPETITION': 'CAT', 'TIME__': 'TIME', 'SCORE H': 'SCORE_H', 'SCORE A': 'SCORE_A'},
               inplace=True)

df_csvs['SCORE_H'] = df_csvs['SCORE_H'].astype('string').str.extract('(\d+)')
df_csvs['SCORE_A'] = df_csvs['SCORE_A'].astype('string').str.extract('(\d+)')

# print(df_csvs)
# print(df_csvs['HOME'].unique())
# print(df_csvs['AWAY'].unique())

