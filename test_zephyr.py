import requests

url = "http://localhost:11434/api/v1/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "zephyr-7b-alpha",
    "prompt": "Hello, how are you?"
}

response = requests.post(url, json=data, headers=headers)

# 检查响应状态码
if response.status_code == 200:
    try:
        json_response = response.json()
        print(json_response.get("completion", "No completion key found"))
    except requests.exceptions.JSONDecodeError as e:
        print("JSON Decode Error:", e)
        print("Raw Response Text:", response.text)
else:
    print("Error:", response.status_code)
    print("Raw Response Text:", response.text)
