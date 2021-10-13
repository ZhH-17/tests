import requests

weaId = 22 # 海淀
date = "2015-07-20"
appkey = "60556"
sign = "da39cb68e4f6ade91cd9619d7e2afb07"

url = f"http://api.k780.com/?app=weather.history&weaId={weaId}&date={date}&appkey={appkey}&sign={sign}&format=json"

res = requests.get(url)
print(res.json())
flag = res.json().get('success')
if flag == '1':
    results = res.json()['result']


'''
https://www.nowapi.com/api/weather.city 请求城市列表
请求示例: http://api.k780.com/?app=weather.city&areaType=cn&appkey=10003&sign=b59bc3ef6191eb9f747dd4e83c99f2a4&format=json
'''
