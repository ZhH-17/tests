# https://www.nowapi.com/api/weather.history
import requests

weaId = 22 # 海淀
date = "2015-07-20"
appkey = "60556"
sign = "da39cb68e4f6ade91cd9619d7e2afb07"

# url = f"http://api.k780.com/?app=weather.history&weaId={weaId}&date={date}&appkey={appkey}&sign={sign}&format=json"

# res = requests.get(url)
# print(res.json())


'''
https://www.nowapi.com/api/weather.city 请求城市列表
请求示例: http://api.k780.com/?app=weather.city&areaType=cn&appkey=10003&sign=b59bc3ef6191eb9f747dd4e83c99f2a4&format=json
'''

import requests
# personally key
key = "0299f708c5144c52ad2ad22f58abf059" 
location = "116.41,39.92"

# grid weather https://dev.qweather.com/docs/api/grid-weather/grid-weather-now/ 
url = f"https://api.qweather.com/v7/grid-weather/now?location={location}&key={key}"
res = requests.get(url)
weather = res.json()

# historical weather https://dev.qweather.com/docs/api/historical/historical-weather/
url = f"https://api.qweather.com/v7/historical/weather?location=101010100&date=20210725&key={key}"
res = requests.get(url)
weather_hourly = res.json()['weatherHourly']


# cma
user = "627624047153xVhyG"
pwd = 'Mp2Kj3D'
t_range = "[20210728000000,20210728140000]"
sta_ids = 'Z9010'
url = f"http://api.data.cma.cn:8090/api?userId={user}&pwd={pwd}&dataFormat=json&interfaceId=getRadaPUPByTimeRangeAndStaID&dataCode=RADA_L3_ST_PRE1H_GIF&timeRange={t_range}&staIds={sta_ids}&elements=Station_Id_C,DATETIME,FORMAT,FILE_NAME"
res = requests.get(url)
data = res.json()
