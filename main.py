import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from process_webscrape import df_comb

# import pandas as pd
# import xgboost as xgb
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.cluster import KMeans  # --/2/--
# # from sklearn.datasets.samples_generator import make_blobs  # --/3/--
# from IPython.display import display
# import matplotlib.pyplot as plt
# from matplotlib.ticker import PercentFormatter, MultipleLocator


df_comb_ssn = df_comb  # [(df_comb['SSN'] == '2019/2020') & (df_comb['CAT'] == 'Div3') & (df_comb['COMP'] == 'LEAGUE')]

df_comb_ssn['DR_'] = df_comb_ssn.groupby(['SSN', 'COMP', 'CAT', 'DR'])['DR'].transform('count')
df_comb_ssn['VT_'] = df_comb_ssn.groupby(['SSN', 'COMP', 'CAT', 'VT'])['VT'].transform('count')
df_comb_ssn['ND_'] = df_comb_ssn.groupby(['SSN', 'COMP', 'CAT', 'ND'])['ND'].transform('count')
df_comb_ssn['CL_'] = df_comb_ssn.groupby(['SSN', 'COMP', 'CAT', 'CL'])['CL'].transform('count')
df_comb_ssn['SC_'] = df_comb_ssn.groupby(['SSN', 'COMP', 'CAT', 'SC'])['SC'].transform('count')
df_comb_ssn['HM_'] = df_comb_ssn.groupby(['SSN', 'COMP', 'CAT', 'HM'])['HM'].transform('count')
df_comb_ssn['NY_'] = df_comb_ssn.groupby(['SSN', 'COMP', 'CAT', 'NY'])['NY'].transform('count')
df_comb_ssn['GS_'] = df_comb_ssn.groupby(['SSN', 'COMP', 'CAT', 'GS'])['GS'].transform('count')
df_comb_ssn['TR_'] = df_comb_ssn.groupby(['SSN', 'COMP', 'CAT', 'TR'])['TR'].transform('count')
df_comb_ssn['UC_'] = df_comb_ssn.groupby(['SSN', 'COMP', 'CAT', 'UC'])['UC'].transform('count')

# print(df_comb_ssn.head(10))

df_test = df_comb_ssn[['DATE', 'SSN', 'COMP', 'CAT', 'DR', 'VT', 'ND', 'CL', 'SC', 'HM', 'NY', 'GS', 'TR', 'UC',
                       'DR_', 'VT_', 'ND_', 'CL_', 'SC_', 'HM_', 'NY_', 'GS_', 'TR_', 'UC_']]
# print(df_test.head())
df_test2 = df_test.melt(id_vars=
                        ['DATE', 'SSN', 'COMP', 'CAT', 'DR', 'VT', 'ND', 'CL', 'SC', 'HM', 'NY', 'GS', 'TR', 'UC'],
                        value_vars=
                        ['DR_', 'VT_', 'ND_', 'CL_', 'SC_', 'HM_', 'NY_', 'GS_', 'TR_', 'UC_'],
                        var_name=
                        'TEAM',
                        value_name=
                        'RES')

df_test2.drop(df_test2[(df_test2.DR == '-') & (df_test2.TEAM == 'DR_')].index, inplace=True)
df_test2.drop(df_test2[(df_test2.VT == '-') & (df_test2.TEAM == 'VT_')].index, inplace=True)
df_test2.drop(df_test2[(df_test2.ND == '-') & (df_test2.TEAM == 'ND_')].index, inplace=True)
df_test2.drop(df_test2[(df_test2.CL == '-') & (df_test2.TEAM == 'CL_')].index, inplace=True)
df_test2.drop(df_test2[(df_test2.SC == '-') & (df_test2.TEAM == 'SC_')].index, inplace=True)
df_test2.drop(df_test2[(df_test2.HM == '-') & (df_test2.TEAM == 'HM_')].index, inplace=True)
df_test2.drop(df_test2[(df_test2.NY == '-') & (df_test2.TEAM == 'NY_')].index, inplace=True)
df_test2.drop(df_test2[(df_test2.GS == '-') & (df_test2.TEAM == 'GS_')].index, inplace=True)
df_test2.drop(df_test2[(df_test2.TR == '-') & (df_test2.TEAM == 'TR_')].index, inplace=True)
df_test2.drop(df_test2[(df_test2.UC == '-') & (df_test2.TEAM == 'UC_')].index, inplace=True)

DR_tag = 'DR'  # 'DROGHEDA'
VT_tag = 'VT'  # 'VINCENTS'
ND_tag = 'ND'  # 'NDWSC'
CL_tag = 'CL'  # 'CLONTARF'
SC_tag = 'SC'  # 'SANDYCOVE'
HM_tag = 'HM'  # 'HALFMOON'
NY_tag = 'NY'  # 'NEWRY'
GS_tag = 'GS'  # 'GUINNESS'
TR_tag = 'TR'  # 'TRINITY'
UC_tag = 'UC'  # 'UCD'

replacement = {
    "DR_": DR_tag,
    "VT_": VT_tag,
    "ND_": ND_tag,
    "CL_": CL_tag,
    "SC_": SC_tag,
    "HM_": HM_tag,
    "NY_": NY_tag,
    "GS_": GS_tag,
    "TR_": TR_tag,
    "UC_": UC_tag,
}

# within the TEAMS filed, replace all instances of "DR_" with "DR", "VT_" with "VT, etc..."
df_test2['TEAM'].replace(replacement, regex=False, inplace=True)


def f(row):
    if row['TEAM'] == DR_tag:
        val = row['DR']
    elif row['TEAM'] == VT_tag:
        val = row['VT']
    elif row['TEAM'] == ND_tag:
        val = row['ND']
    elif row['TEAM'] == CL_tag:
        val = row['CL']
    elif row['TEAM'] == SC_tag:
        val = row['SC']
    elif row['TEAM'] == HM_tag:
        val = row['HM']
    elif row['TEAM'] == NY_tag:
        val = row['NY']
    elif row['TEAM'] == GS_tag:
        val = row['GS']
    elif row['TEAM'] == TR_tag:
        val = row['TR']
    elif row['TEAM'] == UC_tag:
        val = row['UC']
    else:
        val = 'X'
    return val


df_test2['RES_'] = df_test2.apply(f, axis=1)

df_test2['cnt'] = df_test2.groupby(['SSN', 'COMP', 'CAT', 'TEAM'])['TEAM'].transform('count')
# df_test2['uni'] = df_test2.groupby(['SSN', 'COMP', 'CAT', 'TEAM'])['RES_'].transform('nunique')
# df_test2['uni'] = df_test2.groupby(['SSN', 'COMP', 'CAT', 'TEAM'])['TEAM'].agg(['unique'])  # x
# df_test2['uni'] = df_test2.groupby(['SSN', 'COMP', 'CAT', 'TEAM'])['TEAM'].apply(lambda x: list(np.unique(x)))  # x
# df_test2['vct'] = \
#     df_test2.groupby(['SSN', 'COMP', 'CAT', 'TEAM'])['RES_'].transform(lambda x: x.value_counts())
# df.groupby('Item')['Price'].transform(lambda x: x.value_counts().idxmax()))
# data.groupby(['Teacher']).size()
# .groupby(['CAT', 'COMP', 'SSN']).value_counts()['L'])

# df_test3 = df_test2.melt(id_vars=['DATE', 'SSN', 'COMP', 'CAT', 'DR', 'VT', 'ND', 'CL',
#                                   'SC', 'HM', 'NY', 'GS', 'TR', 'UC', 'TEAM', 'RES', 'RES_'],
#                         value_vars=['DR_', 'VT_', 'ND_', 'CL_', 'SC_', 'HM_', 'NY_', 'GS_', 'TR_', 'UC_'],
#                         var_name='TEAM',
#                         value_name='RES')
#
df_test2['WNW'] = np.where(df_test2['RES_'] == 'W', 'WIN', '~WIN')
df_test2['WIN'] = np.where(df_test2['RES_'] == 'W', df_test2['RES'], 0)
df_test2['~WIN'] = df_test2['cnt'] - df_test2['WIN']  # np.where(df_test2['RES_'] != 'W', 1, 0)
df_test3 = df_test2[df_test2['WIN'] != 0]
df_test4 = df_test3.groupby(['SSN', 'COMP', 'CAT', 'TEAM']).take([0])  # df_test2[df_test2['WIN'] != 0]
# df_test2['x_plt'] = df_test2.groupby(['SSN', 'COMP', 'CAT', 'TEAM', 'WNW'])['WIN'].transform('sum')
# df_test2['y_plt'] = df_test2.groupby(['SSN', 'COMP', 'CAT', 'TEAM', 'WNW'])['~WIN'].transform('sum')

# df_test2['uni'] = df_test2.groupby(['SSN', 'COMP', 'CAT', 'TEAM'])['TEAM'].apply(lambda x: list(np.unique(x)))  # x
print(df_test4.shape)
print(df_test4.head(25))
print(type(df_test4))

# -----------//sns plotting//---------------------------------------------------------

sns.set(font_scale=3)
sns.set_style('whitegrid')  # 'darkgrid'

# ----------//1/2: scatter plot of teams//--------------------------------------------------

h = sns.relplot(data=df_test4, x='WIN', y='~WIN', hue='TEAM', s=300)  # , hue_order=_genders, aspect=1.61)

header_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
x_tick_interval = 1
value_tick = range(0, 11, x_tick_interval)
header_name_sel = [header_name[idx] for idx in range(0, len(value_tick))]
for ax in h.axes.flat:
    labels_x = ax.get_xticklabels()  # get x labels
    labels_y = ax.get_yticklabels()  # get y labels
    ax.set_xticks(ticks=value_tick)  # set new labels
    ax.set_yticks(ticks=value_tick)  # set new labels
    ax.set_xticklabels(labels=header_name)  # fontsize=8, rotation=45,
    ax.set_yticklabels(labels=header_name)  # fontsize=8, rotation=45,

# get legend and change stuff
# handles, lables = h.get_legend_handles_labels()
# for handles in handles:
#     handles.set_markersize(10)
#
# # replace legend using handles and labels from above
# lgnd = plt.legend(handles, lables, bbox_to_anchor=(1.02, 1), loc='lower center', borderaxespad=0, title='TITLE')

h.set_xlabels('# Wins')
h.set_ylabels('# Draws/Losses')
plt.title('Non-wins vs. Wins for each team')
# plt.legend(loc='lower center')
plt.show()

# ----------//2/2: stacked bar chart team results over different seasons//------------------

g = sns.displot(data=df_test2, y='TEAM', hue='RES_', col='SSN', multiple='stack', shrink=0.7, palette='PuBuGn',
                kind="hist")
# colors = ['b']
# palettes =['pastel', 'turbo', 'PuBuGn', 'hot']
# multiples =['stack', 'fill']

g.set(xlabel=None)
plt.show()
