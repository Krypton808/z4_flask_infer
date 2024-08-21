import requests
import json

url = 'http://gbox11.aigauss.com:9601/infer'
headers = {
    'Content-Type': 'application/json',
    'Accept': 'text/event-stream'
}

data_2 = {"mrn": "xxx", "series": "xxx"}

r = requests.post(url, headers=headers, json=data_2, stream=True)
r.encoding = 'utf-8'
for line in r.iter_lines(decode_unicode=True):
    print(line)
