import numpy as np
import pandas as pd
import requests as req  # Importing the HTTP library
from dateutil.relativedelta import relativedelta

from process_csvs import df_csvs
import datetime as dt
import re

# link_path = '/Users/duncan/Documents/programming/Python/myPyCharm/UCDPA_Project/fixtures_21-22.html'
# link_path = '/Users/duncan/Documents/college/UCD_PDDA/UCDPA_Project_ACTUAL/fixtures_21-22.html'

# ----------//web scrape//------------------------------------------------------------------------------------------
link_url = 'https://irelandwaterpolo.ie/leinster-league-2021-2022/'
Web = req.get(link_url)  # Requesting for the website
html_tables = pd.read_html(Web.text, header=0)  # extract html content from web page text using readhtml
df_full = html_tables[6]  # extract the 7th table (i.e. index 6)

# ----------//data cleaning//------------------------------------------------------------------------------------------
del df_full['Unnamed: 5']  # drop first irrelevant column
df_full.rename(columns={'SEPT': 'VENUE', 'COMP': 'CAT'}, inplace=True)  # rename columns appropriately
df_dna = df_full.dropna()  # drop null rows

df_dna = df_dna.astype(str)  # convert to str to enable following edits...
pd.set_option('display.max_columns', None)  # enable printing of ALL columns

df_dna.drop(df_dna[~df_dna['VENUE'].isin(['NAC', 'TCD'])].index, inplace=True)  # drop irrelevant rows

df_dna.loc[df_dna.DATE != 'Nan', 'DATE'] = df_dna['DATE'].str.replace('.', '/', regex=False)  # replace "." with "/"
df_dna.loc[df_dna.TIME != 'Nan', 'TIME'] = df_dna['TIME'].str.replace('.', ':', regex=False)  # replace "." with ":"
df_dna.loc[df_dna.TIME != 'Nan', 'TIME'] = df_dna['TIME'] + ':00'  # finish time format "hh:mm:ss"

# add "0" where appropriate to date variable (i.e. change 3/10 to 03/10 and )
df_dna['len_date'] = df_dna['DATE'].apply(lambda x: len(x))
df_dna['len_time'] = df_dna['TIME'].apply(lambda x: len(x))
df_dna.loc[df_dna.len_date == 4, 'DATE'] = '0' + df_dna['DATE']
df_dna.loc[df_dna.len_time == 7, 'TIME'] = '0' + df_dna['TIME']
del df_dna['len_date']
del df_dna['len_time']

# put correct year on end of date
df_dna['test'] = df_dna['DATE'].str[-2:]
df_dna = df_dna.astype({'test': int})
df_dna.loc[df_dna.test <= 6, 'DATE'] = df_dna['DATE'] + '/2022'  # if month between january and june, 2022
df_dna.loc[df_dna.test > 6, 'DATE'] = df_dna['DATE'] + '/2021'  # if month between july and december, 2021
del df_dna['test']

# pd.to_datetime(df_dna['DATE'], format='%d/%m/%Y', errors='ignore')      #orgnl
df_dna['DATE'] = pd.to_datetime(df_dna['DATE'], format="%d/%m/%Y")  # from read csvs

# valid game scores only
df_dna = df_dna[df_dna['SCORE'].str.contains(r'\d{1,2}-\d{1,2}')]

# parse out the scores of each team
df_dna['SCORE_H'] = df_dna['SCORE'].str.extract(r'(\d+)-\d+')
df_dna['SCORE_A'] = df_dna['SCORE'].str.extract(r'\d+-(\d+)')
del df_dna['SCORE']

# -------// v combine 2016-2020 and 2021-2022 data here v //---------------

# df_dna contains 2021-2022 data
# df_csvs contains 2016-2020 data
df_comb = df_dna.append(df_csvs)
# df_comb contains combined data

# >> (482, 8)

# -------// ^ combine 2016-2020 and 2021-2022 data here ^ //---------------

# make league and cup classifications
df_comb.loc[
    df_comb['CAT'].str.contains('cup', case=False) |
    df_comb['CAT'].str.contains('sf', case=False) |
    df_comb['CAT'].str.contains('qf', case=False) |
    df_comb['CAT'].str.contains('final', case=False), 'COMP'] = 'Cup'
df_comb.loc[df_comb['COMP'] != 'Cup', 'COMP'] = 'League'

# print(df_comb.info())
df_comb = df_comb.dropna()
# print(df_comb.info())

# >> (446, 9)

# allocate each outcome as a "H"ome win, an "A"way win or a "D"raw
# df_comb = df_comb.astype({'SCORE_H': int, 'SCORE_A': int})
# df_dna = df_dna.astype({'test': int})

# df_comb['SCORE_H'].astype(str).astype(int)
# df_comb['SCORE_A'].astype(str).astype(int)

# convert score fields to numbers for comparison operators
df_comb['SCORE_H'] = pd.to_numeric(df_comb['SCORE_H'])
df_comb['SCORE_A'] = pd.to_numeric(df_comb['SCORE_A'])

# calculate "FTR" column (i.e., full time result) as win (W), lose (L) or draw (D)
df_comb.loc[df_comb.SCORE_H > df_comb.SCORE_A, 'FTR'] = 'H'
df_comb.loc[df_comb.SCORE_H < df_comb.SCORE_A, 'FTR'] = 'A'
df_comb.loc[df_comb.SCORE_H == df_comb.SCORE_A, 'FTR'] = 'D'

# make fields with team names and competition/category names all upper case
df_comb['HOME'] = df_comb['HOME'].str.upper()
df_comb['AWAY'] = df_comb['AWAY'].str.upper()
df_comb['CAT'] = df_comb['CAT'].str.upper()
df_comb['COMP'] = df_comb['COMP'].str.upper()


def panda_strip(x):
    """
    strip whitespace from every Pandas Data frame cell that has a stringlike object in it
    sourced from:
    https://stackoverflow.com/questions/33788913/pythonic-efficient-way-to-strip-whitespace-from-every-pandas-data-frame-cell-tha
    solution provided by stackoverflow user "Saul Frank" (https://stackoverflow.com/users/2023304/saul-frank)

    :param x: filed or pandas dataframe column to perform whitespace stripping on
    :return: none
    """

    r = []
    for y in x:
        if isinstance(y, str):
            y = y.strip()

        r.append(y)
    return pd.Series(r)


df_comb = df_comb.apply(lambda x: panda_strip(x))

df_comb.replace({'HOME': {'DROGS': 'DROGHEDA', 'STV': 'VINCENTS', 'ST VINCENTS': 'VINCENTS', 'ST. VINCENTS': 'VINCENTS',
                          'TCD': 'TRINITY', 'NTH DUBLIN': 'NDWSC', 'NORTH DUBLIN': 'NDWSC',
                          'HALF MOON': 'HALFMOON', 'HM': 'HALFMOON'}}, regex=False, inplace=True)
df_comb.replace({'AWAY': {'DROGS': 'DROGHEDA', 'STV': 'VINCENTS', 'ST VINCENTS': 'VINCENTS', 'ST. VINCENTS': 'VINCENTS',
                          'TCD': 'TRINITY', 'NTH DUBLIN': 'NDWSC', 'NORTH DUBLIN': 'NDWSC',
                          'HALF MOON': 'HALFMOON', 'HM': 'HALFMOON'}}, regex=False, inplace=True)

df_comb.drop(df_comb[df_comb['AWAY'].isin(['LEINSTER U/19'])].index, inplace=True)  # drop irrelevant rows

# >> (443, 10)

# print(df_comb.sample(20))
# df_test = pd.DataFrame(df_comb, columns=['CAT', 'COMP'])
# print(df_comb.sample(18))

# map all variations of categories to a common string
# e.g., {U13, U13QF, U13 CUP FINAL,...}                => U13M, meaning under 13 mixed
# e.g., {'LLD1', 'D1 MEN CUP SF1', 'DIV 1 S/F 2',...}  => Div1, meaning under Division 1 mens
# etc...
u13mixed_ = ['U13',
             'U13QF',
             'U13 SF1',
             'U13 SF2',
             'U13 CUP FINAL']

u15b_ = ['U15B',
         'U15B SF',
         'U15B FINAL']

u15g_ = ['U15G',
         'U15G FINAL']

u16b_ = ['U16BOYS',
         'BOYS U16',
         'U16 BOYS CUP RD1',
         'U16 BOYS CUP SF1',
         'U16 BOYS CUP SF2',
         'QF1 U16CUPBOYS',
         'QF2 U16CUPBOYS',
         'SF1 U16CUPBOYS',
         'SF2 U16CUPBOYS',
         'U16 BOYS S/F 1',
         'U16 BOYS S/F 2',
         'BOYS U16 SF1',
         'BOYS U16 SF2',
         'U16 BOYS FINAL',
         'BOYS U16 FINAL',
         'U16 BOYS CUP FINAL',
         'U16 SEAN LAWLOR CUP']

u16g_ = ['U16GIRLS',
         'GIRLS U16',
         'SF U16CUPGIRLS',
         'U16GIRLS S/F',
         'GIRLS U16 SF1',
         'U16 GIRLS FINAL',
         'GIRLS U16 FINAL',
         'U16 GIRLS CUP FINAL',
         'U16 GIRLS CUP']

u17b_ = ['U17B',
         'U17B CUP SF1',
         'U17B CUP SF2',
         'U17B / CUP FINAL']

u17g_ = ['U17G',
         'U17G CUP SF',
         'U17G / CUP FINAL']

u19b_ = ['SF1 U19BOYSCUP',
         'SF2 U19BOYSCUP',
         'U19 BOYS S/F 1',
         'U19 BOYS S/F 2',
         'BOYS U19 CUP SF1',
         'BOYS U19 CUP SF2',
         'BOYS U19 CUP SF 1',
         'BOYS U19 CUP SF 2',
         'U19B CUP SF2',
         'U19 BOYS FINAL',
         'BOYS U19 CUP FINAL',
         'U19 BOYS CUP FINAL',
         'U19 ARTHUR DUNNE FINAL']

u19g_ = ['SF U19GIRLSCUP',
         'U19 GIRLS FINAL',
         'U19 GIRLS CUP FINAL']

Div1_ = ['LLD1',
         'D1 MEN CUP SF2',
         'D1 MEN CUP SF1',
         'DIV 1 S/F 1',
         'DIV 1 S/F 2',
         'DIV 1 SF1',
         'DIV 1 SF2',
         'DIV 1 FINAL',
         'SNR MENS FINAL',
         'LEINSTER SENIOR CUP',
         'DIV 1 CUP']

Div2_ = ['LLD2',
         'DIV 2',
         'D2 MEN CUP QF',
         'QF D2CUP',
         'DIV 2 CUP RD1',
         'DIV 2 QF1',
         'DIV 2 SF1',
         'SF1 D2CUP',
         'SF2 D2CUP',
         'D2 MEN CUP SF1',
         'D2 MEN CUP SF2',
         'DIV 2 S/F 1',
         'DIV 2 S/F 2',
         'DIV 2 CUP SF 1',
         'DIV 2 CUP SF 2',
         'MENS D2 FINAL',
         'DIV 2 MENS CUP FINAL',
         'DIV 2 FINAL']

Div3_ = ['LLD3',
         'DIV 3',
         'D3 MEN CUP QF1',
         'D3 MEN CUP QF2',
         'D3 MEN CUP SF1',
         'D3 MEN CUP SF2',
         'D3 MEN CUP QF3',
         'QF D3CUP',
         'DIV 3 CUP RD1',
         'SF1 D3CUP',
         'SF2 D3CUP',
         'DIV 3 S/F 1',
         'DIV 3 S/F 2',
         'DIV 3 SF1',
         'DIV 3 SF2',
         'DIV 3 CUP SF 1',
         'DIV 3 CUP SF 2',
         'MENS D3 FINAL',
         'DIV 3 MENS CUP FINAL',
         'DIV 3 FINAL']

Ladies = ['LADIES CUP SF',
          'LADIES S/F 11',
          'LADIES D1 FINAL',
          'LADIES FRIENDLY',
          'SENIOR LADIES CUP FINAL',
          'LS D1 CUP FINAL',
          'LADIES FINAL',
          'LADIES SENIOR CUP']

df_comb['CAT'].replace(u13mixed_, 'U13M', inplace=True)

df_comb['CAT'].replace(u15b_, 'U15B', inplace=True)
df_comb['CAT'].replace(u15g_, 'U15G', inplace=True)

df_comb['CAT'].replace(u16b_, 'U16B', inplace=True)
df_comb['CAT'].replace(u16g_, 'U16G', inplace=True)

df_comb['CAT'].replace(u17b_, 'U17B', inplace=True)
df_comb['CAT'].replace(u17g_, 'U17G', inplace=True)

df_comb['CAT'].replace(u19b_, 'U19B', inplace=True)
df_comb['CAT'].replace(u19g_, 'U19G', inplace=True)

df_comb['CAT'].replace(Div1_, 'Div1', inplace=True)
df_comb['CAT'].replace(Div2_, 'Div2', inplace=True)
df_comb['CAT'].replace(Div3_, 'Div3', inplace=True)

df_comb['CAT'].replace(Ladies, 'Ladies', inplace=True)


# df_comb['WNR'] =
# df_dna.loc[df_dna.test <= 6, 'DATE'] = df_dna['DATE'] + '/2022'  # if month between january and june, 2022
# df_dna.loc[df_dna.len_date == 4, 'DATE'] = '0' + df_dna['DATE']
# df_comb.loc[df_comb['CAT'], 'WNR'] = 'N'


def flag_df(df, team):
    """
    function to be used to create a new field in the dataframe.
    inserts a "W" for a win, "L" for loss, and "D" for a draw, all based on the score fields.
    Function only applies this to where it is relevant, i.e., it will populate a new "DR"
    field (denoting the team "DROGHEDA") with:
        -   a "W" if they were either the home team or the away team and they won
        -   an "L" if they were either the home team or the away team and they lost
        -   a "D" if they were either the home team or the away team and the game was a tie/draw
        -   a "-" if they were neither the home team or the away team in that game instance

    :param team: the team for which you want to calculate the outcome for (i.e., "W", "L", "D" or "-")
    :param  df: the dataframe to apply function to (function is called as part of the apply method
    to the dataframe, so this argument gets passed automatically
    :return: none
    """

    if df['HOME'] == team:
        if df['SCORE_H'] > df['SCORE_A']:
            return 'W'
        elif df['SCORE_H'] < df['SCORE_A']:
            return 'L'
        else:
            return 'D'
    elif df['AWAY'] == team:
        if df['SCORE_H'] > df['SCORE_A']:
            return 'L'
        elif df['SCORE_H'] < df['SCORE_A']:
            return 'W'
        else:
            return 'D'
    else:
        return '-'


df_comb['DR'] = df_comb.apply(flag_df, args=('DROGHEDA',), axis=1)
df_comb['VT'] = df_comb.apply(flag_df, args=('VINCENTS',), axis=1)
df_comb['ND'] = df_comb.apply(flag_df, args=('NDWSC',), axis=1)
df_comb['CL'] = df_comb.apply(flag_df, args=('CLONTARF',), axis=1)
df_comb['SC'] = df_comb.apply(flag_df, args=('SANDYCOVE',), axis=1)
df_comb['HM'] = df_comb.apply(flag_df, args=('HALFMOON',), axis=1)
df_comb['NY'] = df_comb.apply(flag_df, args=('NEWRY',), axis=1)
df_comb['GS'] = df_comb.apply(flag_df, args=('GUINNESS',), axis=1)
df_comb['TR'] = df_comb.apply(flag_df, args=('TRINITY',), axis=1)
df_comb['UC'] = df_comb.apply(flag_df, args=('UCD',), axis=1)

# calculate a "SSN" (season) field (i.e., "2016/2017", "2019/2020", etc...)
# this was done by calculating a "year" field, a "year plus 1" field, and a "year minus 1" field
df_comb['YEAR'] = pd.DatetimeIndex(df_comb['DATE']).year
df_comb['YEAR_p1'] = pd.DatetimeIndex(df_comb['DATE']).year + 1
df_comb['YEAR_m1'] = pd.DatetimeIndex(df_comb['DATE']).year - 1

df_comb[['YEAR', 'YEAR_p1', 'YEAR_m1']] = df_comb[['YEAR', 'YEAR_p1', 'YEAR_m1']].astype('string')

# need to check whether month is before or after June, i.e., month 6 (June represents the end of the season)
df_comb['SSN'] = np.where(pd.DatetimeIndex(df_comb['DATE']).month > 6
                          , df_comb['YEAR'] + '/' + df_comb['YEAR_p1']  # value if true: append next to current
                          , df_comb['YEAR_m1'] + '/' + df_comb['YEAR']  # value if false: append current to previous
                          )
# we can get rid of helping fields as "SSN" has now been created
df_comb = df_comb.drop(['YEAR', 'YEAR_p1', 'YEAR_m1'], axis=1)

# >> (443, 21)

# df_comb['DATEf64'] = df_comb['DATE'].values.astype("float64")
