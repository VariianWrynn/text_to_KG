from ollama import Client
client = Client(
    host='http://192.168.11.69:11434',
    headers = {"Content-Type": "application/json"}
)
response = client.chat(model='zephyr', messages=[
    {
        'role': 'user',
        'content': 'Why is the sky blue? Be brief.',
    },
])

print(response.message['content'])