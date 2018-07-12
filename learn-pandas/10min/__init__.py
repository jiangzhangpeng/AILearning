# encoding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def test1():
    s = pd.Series([1, 3, 5, np.nan, 6, 8])
    print(s)


def test2():
    dates = pd.date_range('2018-07-10', periods=6)
    month = pd.date_range('2018-07-01', periods=5, freq='M')
    print(dates)
    print(month)

    df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
    print(df)
    print(df.describe())


def test3():
    df = pd.DataFrame(
        {'A': 1.0, 'B': pd.Timestamp('20180708'), 'C': pd.Series(1, index=list(range(4)), dtype='float32'),
         'D': np.array([3] * 4, dtype='int32'), 'E': pd.Categorical(['test', 'train', 'test', 'train']), 'F': 'foo'})
    print(df)
    print(df.describe())
    print(df.dtypes)


def test4():
    dates = pd.date_range('2018-07-10', periods=6)
    df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
    print(df.head(1))
    print(df.tail(1))
    print(df.index)
    print(df.columns)
    print(df.values)
    print(df.T)
    print(df.sort_index(axis=1, ascending=False))
    print(df.sort_values(by='B'))


def test5():
    dates = pd.date_range('2018-07-10', periods=6)
    df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
    print(df['A'])
    print(df[0:2])
    print(df['2018-7-10':'2018-07-12'])


def test6():
    dates = pd.date_range('2018-07-10', periods=6)
    df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
    print(df.loc[dates[0]])
    print(df.loc[dates[0]:dates[2], ['A', 'B']])
    print(df.loc[:, ['A', 'B']])
    print(df.iloc[3:5, 0:2])
    print(df.iloc[[1, 2], [1, 2]])
    print(df.iloc[1, 1])


def test7():
    dates = pd.date_range('2018-07-10', periods=6)
    df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
    print(df[df.A > 0])
    print(df[df > 0])
    df2 = df.copy()
    df2['E'] = ['one', 'two', 'one', 'two', 'three', 'four']
    print(df)
    print(df2)
    print(df2[df2['E'].isin(['one', 'two'])])


def test8():
    dates = pd.date_range('2018-07-10', periods=6)
    df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
    s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('2018-07-13', periods=6))
    df['F'] = s1  # 会匹配s1  series 的index
    print(df)
    df.at[dates[0], 'A'] = 0
    df.iat[0, 1] = 0
    df.loc[:, 'D'] = np.array([5] * len(df))
    print(df)
    df2 = df.copy()
    df2[df2 > 0] = -df2
    print(df2)


def test9():
    dates = pd.date_range('2018-07-10', periods=6)
    df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
    s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('2018-07-13', periods=6))
    df['F'] = s1  # 会匹配s1  series 的index
    df1 = df.reindex(index=dates[0:4], columns=list(df.columns))
    df1['E'] = np.NaN
    df1.iat[2, 4] = 1
    df1.iat[2, 5] = 1
    print(df1)
    print(df1.dropna(how='any'))
    print(df1.fillna(value=66))


def test10():
    dates = pd.date_range('2018-07-10', periods=6)
    df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
    print(df.mean())
    print(df.mean(1))
    print(df)
    print(df.apply(np.cumsum))
    print(df.apply(lambda x: x.max() - x.min()))


def test11():
    s = pd.Series(np.random.randint(0, 7, size=10))
    print(s)
    print(s.value_counts())

    s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
    print(s.str.lower())
    print(s.str.upper())


def test12():
    df = pd.DataFrame(np.random.randn(10, 4))
    print(df)
    df1 = df[0:3]
    df2 = df[3:7]
    df3 = df[7:]
    print(df1)
    print(df2)
    print(df3)
    print(pd.concat([df1, df2, df3]))


def test13():
    left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
    right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
    print(pd.merge(left, right, on='key'))

    left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
    right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
    print(pd.merge(left, right, on='key'))


def test14():
    df = pd.DataFrame(np.random.randn(8, 4), columns=['A', 'B', 'C', 'D'])
    print(df)
    s = df.iloc[3]
    df = df.append(s, ignore_index=True)  # 若为false，则保留原index
    print(df)


def test15():
    df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
                       'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
                       'C': np.random.randn(8), 'D': np.random.randn(8)})
    print(df)
    print(df.groupby('A').sum())
    print(df.groupby('A').count())
    print(df.groupby('A').min())
    print(df.groupby(['A', 'B']).sum())


def test16():
    tuples = list(zip(*[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                        ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]))
    print(tuples)
    index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
    print(index)
    df = pd.DataFrame(np.random.randn(8, 3), index=index, columns=['A', 'B', 'C'])
    print(df)
    df2 = df[0:4]
    stack = df2.stack()
    print(stack)
    print(stack.unstack(0))


def test17():
    df = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 3, 'B': ['A', 'B', 'C'] * 4,
                       'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                       'D': np.random.randn(12), 'E': np.random.randn(12)})
    print(df)
    print(pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C']))


def test18():
    rng = pd.date_range('2018-01-01', periods=100, freq='S')
    print(rng)
    ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
    print(ts)
    print(ts.resample('5Min').sum())

    rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
    ts = pd.Series(np.random.randn(len(rng)), index=rng)
    print(ts)

    ts_utc = ts.tz_localize('UTC')
    print(ts_utc)

    ts_utc1 = ts_utc.tz_convert('US/Eastern')
    print(ts_utc1)


def test19():
    rng = pd.date_range('1/1/2012', periods=5, freq='M')
    print(rng)
    ts = pd.Series(np.random.randn(len(rng)), index=rng)
    print(ts)
    ps = ts.to_period()
    print(ps)
    print(ps.to_timestamp())


def test20():
    prng = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')
    print(prng)
    ts = pd.Series(np.random.randn(len(prng)), prng)
    ts.index = (prng.asfreq('M', 'e') + 1).asfreq('H', 's') + 9
    print(ts.head())


def test21():
    df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "raw_grade": ['a', 'b', 'b', 'a', 'a', 'e']})
    print(df)
    df['grade'] = df['raw_grade'].astype('category')
    print(df['grade'])
    df['grade'].cat.categories = ['very good', 'good', 'very bad']
    print(df['grade'])
    df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
    print(df['grade'])
    print(df.sort_values(by='grade'))
    print(df.groupby('grade').size())


def test22():
    ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
    ts = ts.cumsum()  # 累加
    print(ts.cumsum())
    plt.figure()
    plt.plot(ts)
    plt.show()


def test23():
    ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
    df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=['A', 'B', 'C', 'D'])
    df = df.cumsum()
    plt.figure()
    plt.plot(df)
    plt.show()


def test24():
    ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
    df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=['A', 'B', 'C', 'D'])
    df.to_csv('foo.csv')
    df1 = pd.read_csv('foo.csv')
    print(df1.describe())


def test25():
    ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
    df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=['A', 'B', 'C', 'D'])
    df.to_excel('foo.xlsx', sheet_name='data sheet1')
    df1 = pd.read_excel('foo.xlsx', 'data sheet1', index_col=None, na_value=['NA'])
    print(df1)


if __name__ == '__main__':
    test25()
