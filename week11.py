import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# sns.set(color_codes=True)
# tp = sns.load_dataset('tips')
# sns.regplot(x='total_bill', y='tip', data=tp);
# sns.lmplot(x='total_bill', y='tip', data=tp);
# sns.lmplot(x='size', y='tip', data=tp);
# sns.lmplot(x='size', y='tip', data=tp, x_jitter=.05);
#
# sns.lmplot(x='size', y='tip', data=tp,
#                                     x_estimator=np.mean);
# sns.lmplot(x='total_bill', y='tip', hue='smoker',
#                                    data=tp);
# sns.lmplot(x='total_bill', y='tip', hue='smoker',
#                         data=tp, markers=['o', 'x'], palette='Set1');
# sns.lmplot(x='total_bill', y='tip', hue='smoker',
#                                     col='time', data=tp);
# sns.lmplot(x='total_bill', y='tip', hue='smoker',
#                                    col='time', row='sex', data=tp);
# sns.lmplot(x='size', y='total_bill', hue='day',
#                                    data=tp);
# sns.lmplot(x='size', y='total_bill', hue='day',
#                                    col='day', data=tp);
# plt.show()
#
#
# sns.lmplot(x='size', y='total_bill', hue='day', col='day',
#                                    data=tp, aspect=0.5, x_jitter=0.2);
# sns.lmplot(x='size', y='total_bill', hue='day', col='day', data=tp, aspect=1, x_jitter=0.2, col_wrap=2, height=3);
#
# sns.jointplot(x='total_bill', y='tip', data=tp,
#                 kind='reg');
# sns.pairplot(tp, x_vars=['total_bill', 'size'],
#                       y_vars=['tip'], height=4, aspect=1.2, kind='reg');
#
# plt.show()

# # FacetGrid의 기본생성
sns.set(style='ticks')
tp = sns.load_dataset('tips')
# g = sns.FacetGrid(tp, col='time')
# plt.show()
#
# # Histogram으로 매핑
# g = sns.FacetGrid(tp, col='time')
# g.map(plt.hist, 'tip');
#
# # 산점도로 매핑 + 색상 분류
# g = sns.FacetGrid(tp, col='sex', hue='smoker')
# g.map(plt.scatter, 'total_bill', 'tip', alpha=0.7)
# g.add_legend()
#
# # 행-열로 나눈 플롯 + 점 분산
# g = sns.FacetGrid(tp, row='smoker', col='time',
#                                                margin_titles=True)
# g.map(sns.regplot, 'size', 'total_bill',
#                             color='0.2', fit_reg=False, x_jitter=0.2);
#
# # 막대 그래프 매핑
# g = sns.FacetGrid(tp, col='day', height=3, aspect=0.6)
# g.map(sns.barplot, 'sex', 'total_bill');
#
# # 밀도 플롯 매핑
# cat_index = tp.day.value_counts().index
# print(cat_index)
# g = sns.FacetGrid(tp, col='day', height=1.8,
#                                                aspect=3.8)
# g.map(sns.distplot, 'total_bill', hist=False,
#                             rug=True);
#
# # 사용자 정의 팔레트 + 스타일링
# pal = dict(Lunch='blue', Dinner='red')
# g = sns.FacetGrid(tp, hue='time', palette=pal)
# g.map(plt.scatter, 'total_bill', 'tip', s=100, alpha=0.7,
#                             linewidth=0.5, edgecolor='black')
#
# plt.show()


df = sns.load_dataset('attention').query('subject<=12')
print(df)
# 서브 플롯 생성
g = sns.FacetGrid(df, col='subject', col_wrap=4,
                                                height=2, ylim=(0, 10))
g.map(sns.pointplot, 'solutions', 'score',
                            color='blue', ci=None);
with sns.axes_style('dark'):
    g = sns.FacetGrid(tp, row='sex', col='smoker', margin_titles=True, height=2.7)
g.map(plt.scatter, 'total_bill', 'tip', color='#475157', edgecolor='orange', lw=1.0);
g.set_axis_labels('Total Bill in US$', 'TIP');
g.set(xticks=[15, 30, 45, 60], yticks=[2, 5, 8, 11]);
g.fig.subplots_adjust(wspace=.1, hspace=.2);
g = sns.FacetGrid(tp, col='smoker', margin_titles=True, height=4.5)
g.map(plt.scatter, 'total_bill', 'tip', color='#009900', edgecolor='red', s=50, lw=.8)
for ax in g.axes.flat:
    ax.plot((0, 50), (0, 12), c='green', ls='--')
g.set(xlim=(0, 60), ylim=(0, 14));
g.axes
print(type(g.axes))
g.axes.flat

plt.show()


