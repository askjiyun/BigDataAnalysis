import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# sns.set()
# print(sns.get_dataset_names())
# tips = sns.load_dataset('tips')
# print(tips)
# #sns.relplot(x='total_bill', y='tip', col='time', hue='smoker', style='smoker', size='size', data=tips)
# sns.relplot(x='total_bill', y='tip', col='day', hue='smoker', style='smoker', size='size', data=tips)
# #plt.show()
#
#
# dots = sns.load_dataset('dots')
# print(dots)
# # 복잡한 데이터 관계를 다차원적 시각화
# sns.relplot(x='time', y='firing_rate', col='align', hue='choice', size='coherence', style='choice',
#                  facet_kws=dict(sharex=False), kind='line', legend='full', data=dots);
# #plt.show()
# #
# x = np.random.normal(size=100)
# # 히스토그램 및 커널 밀도 추정 (KDE) 으로 시각화
# sns.distplot(x,hist=False, rug=True)
#plt.show()

# #이변량 데이터 시각화
# mean, cov = [0, 1], [(1, .5), (.5, 1)]
# data = np.random.multivariate_normal(mean, cov, 300)
# df = pd.DataFrame(data, columns=['x', 'y'])
# #
# print(df)
# # 이변량 시각화
# sns.jointplot(x='x', y='y', data=df);
# #plt.show()
#
# x, y = np.random.multivariate_normal(mean, cov, 1000).T
# with sns.axes_style('white'):
#   sns.jointplot(x=x, y=y, kind='hex', color='b');
# #plt.show()
#
# sns.jointplot(x='x', y='y', data=df, kind='kde');
# #plt.show()

## 시각화
sns.set(style='ticks', color_codes=True)
tp = sns.load_dataset('tips')
g = sns.JointGrid(x='total_bill', y='tip', data=tp)

#plt.show()

# 선형 회귀 플롯 - regplot
sns.set(color_codes=True)
tp = sns.load_dataset('tips')
ax = sns.regplot(x='total_bill', y='tip', data=tp)
#plt.show()

##다변량 정규분포 데이터로 회귀 플롯 - multivariate_normal
# np.random.seed(112)
# mean, cov = [2, 3], [(1.5, 0.6), (0.6, 1)]
# x, y = np.random.multivariate_normal(mean, cov, 30).T
# ax = sns.regplot(x=x, y=y, color='g')
# plt.show()
#
# ax = sns.regplot(x=x, y=y, ci=68)
# plt.show()
#
#
# ax = sns.regplot(x='size', y='total_bill', data=tp, x_jitter=0.1)
# plt.show()
# ax = sns.regplot(x='size', y='total_bill', data=tp, x_estimator=np.mean)
# plt.show()
# tp['big_tip'] = (tp.tip/tp.total_bill) > 0.175
# ax = sns.regplot(x='total_bill', y='big_tip', data=tp,
#                                            logistic=True, n_boot=500, y_jitter=.03)
# plt.show()
#
#
# ax = sns.regplot(x='size', y='total_bill', data=tp, x_estimator=np.mean,
#                                           logx=True, truncate=True)
# #plt.show()


ans = sns.load_dataset('anscombe')
ans[0:3]


ax = sns.regplot(x='x', y='y', scatter_kws={'s': 200},
                                             data=ans.loc[ans.dataset == 'II'],
                                             order=2, ci=None, truncate=True)
# plt.show()
#
# sns.set()
#
# #기본 산점도 : sns.scatterplot()
# tips = sns.load_dataset('tips')
# ax = sns.scatterplot(x='total_bill', y='tip', data=tips)
# plt.show()
#
# ax = sns.scatterplot(x='total_bill', y='tip',
#                           hue='size', size='size', data=tips)
# #plt.show()
#
# ax = sns.scatterplot(x='total_bill', y='tip',
#                                     hue='day', style='time', data=tips)
# #plt.show()
#
# ax = sns.scatterplot(x='total_bill', y='tip', size='size', data=tips)
#
# #plt.show()
#
#
# sns.set()
# fm = sns.load_dataset('fmri')
# fm.head()
# ax = sns.lineplot(x='timepoint', y='signal',data=fm)
# plt.show()
# ax = sns.lineplot(x='timepoint', y='signal',hue='event', data=fm)
# plt.show()
# ax = sns.lineplot(x='timepoint', y='signal',hue='region', style='event', data=fm)
# plt.show()
# ax = sns.lineplot(x='timepoint', y='signal',hue='event', style='event', markers=True,dashes=False, data=fm)
# plt.show()
# ax = sns.lineplot(x='timepoint', y='signal', hue='event', err_style='bars', ci=68, data=fm)
# plt.show()
#
## Catplot 범주형 데이터 시각화
tip = sns.load_dataset('tips'); tip[0:3]
# sns.catplot(x='day', y='total_bill', data=tip);
# plt.show()
# sns.catplot(x='day', y='total_bill', jitter=False,  data=tip)
# plt.show()
# sns.catplot(x='day', y='total_bill',
#                                     kind='swarm', data=tip);
# plt.show()
# sns.catplot(x='day', y='total_bill', hue='sex',
#                                     kind='swarm', data=tip);
# plt.show()
# sns.catplot(x='total_bill', y='day', hue='time',
#                                     kind='swarm', data=tip);
# plt.show()
tip = sns.load_dataset('tips')
sns.catplot(x='day', y='total_bill', kind='box', data=tip);
plt.show()
sns.catplot(x='day', y='total_bill', hue='smoker', kind='box', data=tip)
plt.show()
