import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pdb


def get_data_from_cma(date1, date2, field='prec'):
    # format = YYYYMMDDHHMMSS
    user = "627630863033dLDet"
    pwd = 'l6x6NCt'
    t_range = f"[{date1},{date2}]"
    sta_ids = "54399"
    if field == 'prec':
        field = "PRE_1h"
    eles = "PRE_1h"
    url = f"http://api.data.cma.cn:8090/api?userId={user}&pwd={pwd}&dataFormat=json&interfaceId=getSurfEleByTimeRangeAndStaID&dataCode=SURF_CHN_MUL_HOR&timeRange={t_range}&staIDs={sta_ids}&elements=Year,Mon,Day,Hour,TEM,PRE_1h"
    res = requests.get(url)
    data = res.json()['DS']

    data_key = []
    for item in data:
        t = "%04d-%02d-%02dT%02d:00" \
            % (int(item['Year']), int(item['Mon']), int(item['Day']), int(item['Hour']))
        data_key.append([t, float(item[field])])
    df = pd.DataFrame(data_key, columns=['time', field])
    return data, df


def get_day_data_from_qw(date, field='prec'):
    key = "0299f708c5144c52ad2ad22f58abf059" 
    location = "101010200" # haidian
    print(date)

    url = f"https://api.qweather.com/v7/historical/weather?location={location}&date={date}&key={key}"
    res = requests.get(url)
    weather_hourly = res.json()['weatherHourly']
    data_key = []
    if field == 'prec':
        field = "precip"
    for item in weather_hourly:
        data_key.append([item['time'][:16], float(item[field])])
    df = pd.DataFrame(data_key, columns=['time', field])
    return weather_hourly, df

def get_data_from_qw(date1, date2, field='prec'):
    day1 = int(date1[6:8])
    day2 = int(date2[6:8])
    dfs = []
    for day in range(day1, day2+1):
        date = "202107%02d" % day
        print(date)
        data, df_key = get_day_data_from_qw(date, field)
        dfs.append(df_key)
    df = pd.concat(dfs, axis=0)

    return df


if __name__ == "__main__":
    date1 = "202107%02d000000" % 26
    date2 = "202107%02d235900" % 31
    data1, data1_key = get_data_from_cma(date1, date2, 'prec')
    data1_key.index = data1_key['time']
    data1_key.drop('time', inplace=True, axis=1)

    # data2, data2_key = get_day_data_from_qw(date1, 'prec')
    data2_key = get_data_from_qw(date1, date2, 'prec')
    data2_key.index = data2_key['time']
    data2_key.drop('time', inplace=True, axis=1)

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(data1_key.values.reshape(-1))
    # axs[0].set_xticklabels(
    #     list(map(lambda x: x[5:-3], data1_key.index[::12])), rotation=0)
    axs[0].set_title("data from cma")

    axs[1].plot(data2_key.values.reshape(-1))
    # axs[1].set_xticks(np.arange(0, len(data2_key), 12))
    # axs[1].set_xticklabels(
    #     list(map(lambda x: x[5:-3], data2_key.index[::12])), rotation=0)
    axs[1].set_title("data from hefeng")
    plt.tight_layout()
    cor = np.corrcoef(data1_key.values.reshape(-1),
                      data2_key.values.reshape(-1))[1, 0]
    plt.title('corr coef is %.2f' % cor)
    data1_key.to_csv("data_cmd.csv")
    data2_key.to_csv("data_qw.csv")

    plt.show()
