### Example File to interact with FAST API XGBClassifier model efficiently

import requests

url = "http://localhost:9696/predict"

payload = {
    "age": 73.0,
    "bp": 70.0,
    "sg": 1.005,
    "al": 0.0,
    "su": 0.0,
    "pcc": "notpresent",
    "ba": "notpresent",
    "bgr": 70.0,
    "bu": 32.0,
    "sc": 0.9,
    "sod": 125.0,
    "pot": 4.0,
    "hemo": 10.0,
    "pcv": 29.0,
    "wbcc": 18900.0,
    "rbcc": 3.5,
    "htn": "yes",
    "dm": "yes",
    "cad": "no",
    "appet": "good",
    "pe": "yes",
    "ane": "no"
}

response = requests.post(url, json=payload)

print(response.json())