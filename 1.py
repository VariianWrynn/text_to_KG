from ollama import Client

# 定义 SYS_PROMPT 和 USER_PROMPT
SYS_PROMPT = (
    "你是一个网络图制作者，从给定的上下文中提取术语及其关系。\n"
    "你被提供了一个上下文块（由````限定）你的任务是提取所给上下文中提到的术语的本体。\n"
    "这些术语应该抽提上下文代表关键概念。\n"
    "想法1: 在遍历每个句子时，思考其中提到的关键术语。\n"
    "术语可能包括对象、实体、位置、组织、人物、\n"
    "任务、首字母缩写词、文档、服务、概念等。\n"
    "术语应不可用原子化认。\n"
    "想法2: 思考这些术语如何与其他术语一对一地关联。\n"
    "过滤程序在句子或同一段落中提到的术语彼此相关。\n"
    "术语可以与许多其他术语相相识。\n"
    "想法3: 找出每对这样样的术语之间的关系。\n\n"
    "将你的输出格式化为json列表。列表的每个元素包含一对术语\n"
    "和它们之间的关系，如下所示：\n"
    "[\n"
    "  {\n"
    "    \"node_1\": \"从提取的本体中提取的一个概念\",\n"
    "    \"node_2\": \"从提取的本体中提取的相关概念\",\n"
    "    \"edge\": \"两个概念之间的关系node_1和node_2用一个句子描述\"\n"
    "  },\n"
    "  {...}\n"
    "]"
)

USER_PROMPT = "context: ```{input}``` \n\n output:"

# 替换成你的上下文
input_text = "Why is the sky blue?"  # 示例上下文
formatted_user_prompt = USER_PROMPT.format(input=input_text)

# 创建 Ollama 客户端
client = Client(
    host='http://192.168.11.69:11434',
    headers={"Content-Type": "application/json"}
)

# 使用指定的提示词进行聊天
response = client.chat(model='zephyr', messages=[
    {
        'role': 'system',
        'content': SYS_PROMPT,
    },
    {
        'role': 'user',
        'content': formatted_user_prompt,
    },
])

# 打印响应
print("#########################RESPONSE#########################")
print(response.message['content'])
