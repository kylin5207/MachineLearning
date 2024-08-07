#!/usr/bin/python
# -*- encoding: utf-8
"""
使用pandas读取数据集
"""
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


def enum_row(row):
    print(row['state'])


def find_state_code(row):
    if row['state'] != 0:
        print(process.extractOne(row['state'], states, score_cutoff=80))


def capital(str):
    return str.capitalize()


def correct_state(row):
    if row['state'] != 0:
        state = process.extractOne(row['state'], states, score_cutoff=80)
        if state:
            state_name = state[0]
            return ' '.join(map(capital, state_name.split(' ')))
    return row['state']


def fill_state_code(row):
    if row['state'] != 0:
        state = process.extractOne(row['state'], states, score_cutoff=80)
        if state:
            state_name = state[0]
            return state_to_code[state_name]
    return ''


if __name__ == "__main__":
    # 一些显示设置
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 9)

    # 1. 读取数据
    data = pd.read_excel('.//sales.xlsx', sheet_name='sheet1', header=0)
    # 前5行
    print('data.head() = \n', data.head())
    # 后5行
    print('data.tail() = \n', data.tail())
    # 每列类型，除了数据类型外，其他类型都是object，注意字符串是object类型
    print('data.dtypes = \n', data.dtypes)
    # 列名
    print('data.columns = \n', data.columns)
    for c in data.columns:
        print(c, end=' ')
    print()

    # 生成新列total
    data['total'] = data['Jan'] + data['Feb'] + data['Mar']

    print(data.head())
    # 对某一列进行统计分析
    print("Sum of Jan : ", data['Jan'].sum())
    print("Min of Jan : ", data['Jan'].min())
    print("Max of Jan : ", data['Jan'].max())
    print("Mean of Jan : ", data['Jan'].mean())

    print('=============')
    # 添加一行，计算每个月份的总和
    s1 = data[['Jan', 'Feb', 'Mar', 'total']].sum()
    print("Month Sum :\n", s1)

    s2 = pd.DataFrame(data=s1)
    print(s2)
    print(s2.T)
    # 重新设定列名，输出
    print(s2.T.reindex(columns=data.columns))
    # 即：
    s = pd.DataFrame(data=data[['Jan', 'Feb', 'Mar', 'total']].sum()).T
    s = s.reindex(columns=data.columns, fill_value=0)
    print(s)
    # 在后面加入行
    data = data.append(s, ignore_index=True)
    data = data.rename(index={15: 'Total'})
    print(data.tail())
    print("New data:\n", data)

    # apply的使用
    print('==============apply的使用==========')
    data.apply(enum_row, axis=1)

    state_to_code = {"VERMONT": "VT", "GEORGIA": "GA", "IOWA": "IA", "Armed Forces Pacific": "AP", "GUAM": "GU",
                     "KANSAS": "KS", "FLORIDA": "FL", "AMERICAN SAMOA": "AS", "NORTH CAROLINA": "NC", "HAWAII": "HI",
                     "NEW YORK": "NY", "CALIFORNIA": "CA", "ALABAMA": "AL", "IDAHO": "ID",
                     "FEDERATED STATES OF MICRONESIA": "FM",
                     "Armed Forces Americas": "AA", "DELAWARE": "DE", "ALASKA": "AK", "ILLINOIS": "IL",
                     "Armed Forces Africa": "AE", "SOUTH DAKOTA": "SD", "CONNECTICUT": "CT", "MONTANA": "MT",
                     "MASSACHUSETTS": "MA",
                     "PUERTO RICO": "PR", "Armed Forces Canada": "AE", "NEW HAMPSHIRE": "NH", "MARYLAND": "MD",
                     "NEW MEXICO": "NM",
                     "MISSISSIPPI": "MS", "TENNESSEE": "TN", "PALAU": "PW", "COLORADO": "CO",
                     "Armed Forces Middle East": "AE",
                     "NEW JERSEY": "NJ", "UTAH": "UT", "MICHIGAN": "MI", "WEST VIRGINIA": "WV", "WASHINGTON": "WA",
                     "MINNESOTA": "MN", "OREGON": "OR", "VIRGINIA": "VA", "VIRGIN ISLANDS": "VI",
                     "MARSHALL ISLANDS": "MH",
                     "WYOMING": "WY", "OHIO": "OH", "SOUTH CAROLINA": "SC", "INDIANA": "IN", "NEVADA": "NV",
                     "LOUISIANA": "LA",
                     "NORTHERN MARIANA ISLANDS": "MP", "NEBRASKA": "NE", "ARIZONA": "AZ", "WISCONSIN": "WI",
                     "NORTH DAKOTA": "ND",
                     "Armed Forces Europe": "AE", "PENNSYLVANIA": "PA", "OKLAHOMA": "OK", "KENTUCKY": "KY",
                     "RHODE ISLAND": "RI",
                     "DISTRICT OF COLUMBIA": "DC", "ARKANSAS": "AR", "MISSOURI": "MO", "TEXAS": "TX", "MAINE": "ME"}
    states = list(state_to_code.keys())
    print("fuzz.ratio('Python Package', 'Python Package') = ", fuzz.ratio('Python Package', 'Python Package'))
    print("fuzz.ratio('Python Package', 'PythonPackage') = ", fuzz.ratio('Python Package', 'PythonPackage'))
    print("process.extract('Mississippi', states) = ", process.extract('Mississippi', states))
    print("process.extract('Mississipi', states, limit=1) = ", process.extract('Mississipi', states, limit=1))
    print("process.extractOne('Mississipi', states) = ", process.extractOne('Mississipi', states))

    # 找到州编码
    data.apply(find_state_code, axis=1)

    print('Before Correct State:\n', data['state'])
    data['state'] = data.apply(correct_state, axis=1)
    print('After Correct State:\n', data['state'])
    # 插入一列
    data.insert(5, 'State Code', np.nan)
    data['State Code'] = data.apply(fill_state_code, axis=1)
    print(data)

    # group by
    print('==============group by================')
    print(data.groupby('State Code'))
    print('All Columns:\n')
    print(data.groupby('State Code').sum())
    print('Short Columns:\n')
    print(data[['State Code', 'Jan', 'Feb', 'Mar', 'total']].groupby('State Code').sum())

    # 写入文件
    data.to_excel('sales_result.xls', sheet_name='Sheet1', index=False)
