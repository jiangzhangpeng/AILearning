# encoding:utf-8
import functools
import itertools

import numpy as np
import pandas as pd


def test1():
    df = pd.DataFrame({'AAA': [4, 5, 6, 7], 'BBB': [10, 20, 30, 40], 'CCC': [100, 50, -30, -50]})
    print(df)
    df.loc[df.AAA >= 5, 'BBB'] = -1
    print(df)
    df.loc[df.AAA < 5, ['BBB', 'CCC']] = 2000
    print(df)
    df_mask = pd.DataFrame({'AAA': [True] * 4, 'BBB': [False] * 4, 'CCC': [True, False] * 2})
    print(df.where(df_mask, -1000))


def test2():
    df = pd.DataFrame({'AAA': [4, 5, 6, 7], 'BBB': [10, 20, 30, 40], 'CCC': [100, 50, -30, -50]})
    print(df)
    df['logic'] = np.where(df.AAA >= 5, 'high', 'low')
    print(df)


def test3():
    df = pd.DataFrame({'AAA': [4, 5, 6, 7], 'BBB': [10, 20, 30, 40], 'CCC': [100, 50, -30, -50]})
    print(df)
    dflow = df[df.AAA <= 5]
    dfhigh = df[df.AAA > 5]
    print(dflow)
    print(dfhigh)


def test4():
    df = pd.DataFrame({'AAA': [4, 5, 6, 7], 'BBB': [10, 20, 30, 40], 'CCC': [100, 50, -30, -50]})
    newseries = df.loc[(df['BBB'] < 25) & (df['CCC'] >= -40), ['AAA', 'BBB']]
    print(newseries)
    df.loc[(df['BBB'] > 25) | (df['CCC'] >= 75), 'AAA'] = 0.1
    print(df)


def test5():
    df = pd.DataFrame({'AAA': [4, 5, 6, 7], 'BBB': [10, 20, 30, 40], 'CCC': [100, 50, -30, -50]})
    aValue = 43.0
    df = df.loc[(df.CCC - aValue).abs().argsort()]
    print(df)
    print((df.CCC - aValue).abs().argsort())


def test6():
    df = pd.DataFrame({'AAA': [4, 5, 6, 7], 'BBB': [10, 20, 30, 40], 'CCC': [100, 50, -30, -50]})
    print(df[(df.AAA <= 6) & (df.index.isin([0, 2, 4]))])


def test7():
    data = {'AAA': [4, 5, 6, 7], 'BBB': [10, 20, 30, 40], 'CCC': [100, 50, -30, -50]}
    df = pd.DataFrame(data=data, index=['foo', 'bar', 'boo', 'kar'])
    print(df)
    print(df.loc['foo':'boo'])
    print(df.iloc[0:3])


def test8():
    rng = pd.date_range('1/1/2013', periods=100, freq='D')
    data = np.random.randn(100, 4)
    cols = ['A', 'B', 'C', 'D']
    df1, df2, df3 = pd.DataFrame(data=data, index=rng, columns=cols), \
                    pd.DataFrame(data=data, index=rng, columns=cols), \
                    pd.DataFrame(data=data, index=rng, columns=cols)

    # print(df1)
    pf = pd.Panel({'df1': df1, 'df2': df2, 'df3': df3})
    print(pf)
    pf.loc[:, :, 'F'] = pd.DataFrame(data, rng, cols)
    print(pf)


def test9():
    df = pd.DataFrame({'AAA': [1, 2, 1, 3], 'BBB': [1, 1, 2, 2], 'CCC': [2, 1, 3, 1]})
    print(df)
    source_cols = df.columns
    print(source_cols)
    new_cols = [str(x) + '_cat' for x in source_cols]
    print(new_cols)
    categories = {1: 'Alpha', 2: 'Beta', 3: 'Charlie'}
    df[new_cols] = df[source_cols].applymap(categories.get)
    print(df)
    df = df[source_cols].applymap(categories.get)
    print(df)


def test10():
    df = pd.DataFrame({'AAA': [1, 1, 1, 2, 2, 2, 3, 3], 'BBB': [2, 1, 3, 4, 5, 1, 2, 3]})
    print(df)
    t = df.loc[df.groupby('AAA')['BBB'].idxmin()]
    print(t)
    tt = df.sort_values(by="BBB").groupby("AAA", as_index=False).first()
    print(tt)


def test11():
    df = pd.DataFrame(
        {'row': [0, 1, 2], 'One_X': [1.1, 1.1, 1.1], 'One_Y': [1.2, 1.2, 1.2], 'Two_X': [1.11, 1.11, 1.11],
         'Two_Y': [1.22, 1.22, 1.22]})
    print(df)
    df = df.set_index('row')
    print(df)
    df.columns = pd.MultiIndex.from_tuples([tuple(c.split('_')) for c in df.columns])
    print(df)
    # df.to_excel('column_transform.xlsx')
    df = df.stack(0).reset_index(1)
    print(df)


def test12():
    cols = pd.MultiIndex.from_tuples([(x, y) for x in ['A', 'B', 'C'] for y in ['O', 'I']])
    df = pd.DataFrame(np.random.randn(2, 6), index=['n', 'm'], columns=cols)
    print(df)
    df = df.div(df['C'], level=1)
    print(df)


def test13():
    coords = [('AA', 'one'), ('AA', 'six'), ('BB', 'one'), ('BB', 'two'), ('BB', 'six')]
    index = pd.MultiIndex.from_tuples(coords)
    df = pd.DataFrame([11, 22, 33, 44, 55], index, ['MyData'])
    print(df)

    t1 = df.xs('BB', level=0, axis=0)
    print(t1)

    t2 = df.xs('six', level=1, axis=0)
    print(t2)


def test14():
    index = list(itertools.product(['Ada', 'Quinn', 'Violet'], ['Comp', 'Math', 'Sci']))
    headr = list(itertools.product(['Exams', 'Labs'], ['I', 'II']))
    indx = pd.MultiIndex.from_tuples(index, names=['Student', 'Course'])
    cols = pd.MultiIndex.from_tuples(headr)  # Notice these are un-named
    data = [[70 + x + y + (x * y) % 3 for x in range(4)] for y in range(9)]
    df = pd.DataFrame(data, indx, cols);
    print(df)
    All = slice(None)
    print(df.loc['Violet'])
    print(df.loc[(All, ['Math', 'Sci']), ('Exams', 'I')])
    df = df.sort_values(by=('Labs', 'II'), ascending=False)
    print(df)


def test15():
    df = pd.DataFrame(np.random.randn(6, 1), index=pd.date_range('2013-08-01', periods=6, freq='B'), columns=list('A'))
    print(df)
    df.iloc[3, 0] = np.NaN  # 这两种方式等价，下面的更易于理解
    df.loc[df.index[3], 'A'] = np.nan
    print(df)
    df = df.reindex(df.index[::-1]).ffill()
    print(df)


def GrowUp(x):
    avg_weight = sum(x[x['size'] == 'S'].weight * 1.5)
    avg_weight += sum(x[x['size'] == 'M'].weight * 1.25)
    avg_weight += sum(x[x['size'] == 'L'].weight)
    avg_weight /= len(x)
    return pd.Series(['L', avg_weight, True], index=['size', 'weight', 'adult'])


def test16():
    df = pd.DataFrame({'animal': 'cat dog cat fish dog cat cat'.split(),
                       'size': list('SSMMMLL'),
                       'weight': [8, 10, 11, 1, 20, 12, 12],
                       'adult': [False] * 5 + [True] * 2});
    print(df)
    t = df.groupby('animal').apply(lambda subf: subf['size'][subf['weight'].idxmax()])
    print(t)
    gb = df.groupby('animal')
    print(gb.get_group('cat'))
    expect_df = gb.apply(GrowUp)
    print(expect_df)


def CumRet(x, y):
    return x * (1 + y)


def Red(x):
    return functools.reduce(CumRet, x, 1.0)


def test17():
    S = pd.Series([i / 100.0 for i in range(1, 11)])
    print(S)


if __name__ == '__main__':
    test17()
