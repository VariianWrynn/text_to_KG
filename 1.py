import sys
from ollama import Client

# 定义 SYS_PROMPT 和 USER_PROMPT
SYS_PROMPT = (
    "你是一个网络图制作者，从给定的上下文中提取术语及其关系。\n"
    "你被提供了一个上下文块（由````限定）你的任务是提取所给上下文中提到的术语的本体。\n"
    
    "思路1: 在遍历每个句子时，思考其中提到的关键术语。 术语可能包括对象、实体、位置、组织、人物、任务、首字母缩写词、文档、服务、概念等。\n"
    "术语类别: \n"
    "人物：如John，Mary。\n"
    "地点：如Paris，New York。\n"
    "组织：如Google，NASA。\n"
    "事件/活动：如会议，讲座。\n"
    "物品/物体：如手机，汽车。\n"
    "概念/抽象术语：如自由，科技。\n"
    "时间/日期：如2022年5月，昨天。\n"
    "任务：如进行调查，提交报告。\n"
    "文档：如项目报告，研究论文。\n"
    "其他术语：如计算机程序，网络协议。\n"

    "思路2: 思考这些术语如何与其他术语一对一地关联。 过滤程序在句子或同一段落中提到的术语彼此相关。 术语可以与许多其他术语相相识。\n"
    "关系类型：\n"
    "拥有：如John owns a car。\n"
    "属于：如The book belongs to Mary。\n"
    "参与：如Alice participated in the event。\n"
    "位置：如Paris is in France。\n"
    "所属组织：如John works for Microsoft。\n"
    "因果关系：如The rain caused the flood。\n"
    "时间关系：如The meeting is scheduled for tomorrow。\n"
    "上下文关系：如Mary met John at the event。\n"
    "其他复杂关系：如This event happened due to a change in policy。\n"

    "思路3: 找出每对术语之间的关系。 对于复杂或模糊的关系，考虑根据上下文推理。例如，如果文本中提到“Mary owns a car”，你可以推测“Mary”和“car”之间存在“拥有”关系。\n\n"
    "跨句和跨段落关系：\n"
    "在长文本中，术语的关系可能跨越多个句子或段落。请注意，跨句、跨段落的术语间可能存在直接或间接的关系。\n"
    "如果术语在多个句子中提到，并且它们之间有联系，考虑它们可能的关系。\n"
    "和它们之间的关系，如下所示：\n"

    "歧义处理：\n"
    "对于歧义术语，根据上下文来推测其具体含义。\n"
    "例如，“Apple”可能指公司，也可能指水果；“Bank”可能指金融机构或河岸。\n"

    "将你的输出格式化为 JSON 列表。列表的每个元素包含一对术语和它们之间的关系，如下所示：\n"
    "[\n"
    "  {\n"
    "    \"node_1\": \"术语1\",\n"
    "    \"node_2\": \"术语2\",\n"
    "    \"edge\": \"术语1和术语2之间的关系描述\"\n"
    "  },\n"
    "  {\n"
    "    \"node_1\": \"术语3\",\n"
    "    \"node_2\": \"术语4\",\n"
    "    \"edge\": \"术语3和术语4之间的关系描述\"\n"
    "  },\n"
    "  {...}\n"
    "]"
)

USER_PROMPT = "context: ```{input}``` \n\n output:"

# 获取命令行参数作为输入文本
if len(sys.argv) > 1:
    input_text = sys.argv[1]
else:
    print("警告: 未提供输入参数，使用默认输入文本。")
    input_text = "Mary had a little lamb, You've heard this story before; But did you know she passed her plate, And ate a little more!"  # 示例上下文

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
