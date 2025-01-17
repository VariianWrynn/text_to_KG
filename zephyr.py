import sys
from ollama import Client
import json
import pandas as pd

client = Client(
    host='http://192.168.11.69:11434',
    headers={"Content-Type": "application/json"}
)
"""
SYS_PROMPT = (
    "你是一个网络图制作者，从给定的上下文中提取术语及其关系。你的目标是生成一个清晰且简洁的知识图谱，并优先归一化相似的术语对。\n"
    "你被提供了一个上下文块（由````限定）你的任务是提取所给上下文中提到的术语的本体。\n"
    
    "思路1: 在遍历每个句子时，提取提到的关键术语。术语可能包括对象、实体、概念、任务等。考虑术语的含义和上下文来识别相同含义的不同表达方式。\n"
    "术语归一化示例: \n"
    "“AI”和“人工智能”应视为相同术语。\n"
    "“Fraud detection system”和“Fraud detection”可以归类为同一术语。\n"
    "“Healthcare”和“Medical field”可以归一化为“Healthcare”。\n"
    "实体类别实例：\n"
    "事件/活动：如会议，讲座。\n"
    "物品/物体：如手机，汽车。\n"
    "概念/抽象术语：如自由，科技。\n"
    "时间/日期：如2022年5月，昨天。\n"
    "任务：如进行调查，提交报告。\n"
    "文档：如项目报告，研究论文。\n"
    "其他术语：如计算机程序，网络协议。\n"

    "思路2: 提取术语之间的关系，并优先合并表达相似含义的关系。例如：\n"
    "相似含义类型：\n"
    "“is applied in”和“used in”可以统一为“applied in”。\n"
    "“is related to”和“connected to”可以统一为“related to”。"
    "复杂关系可以根据上下文推断其含义。例如，如果文本中提到“Mary owns a car”，你可以推测“Mary”和“car”之间存在“拥有”关系。\n\n"
    "关系类型：\n"
    "拥有：如John owns a car。\n"
    "时间关系：如The meeting is scheduled for tomorrow。\n"
    "上下文关系：如Mary met John at the event。\n"
    "其他复杂关系：如This event happened due to a change in policy。\n"

    "思路3: 找出每对术语之间的关系。 对于复杂或模糊的关系，考虑根据上下文推理。例如，如果文本中提到“Mary owns a car”，你可以推测“Mary”和“car”之间存在“拥有”关系。\n\n"
    "跨句和跨段落关系：\n"
    "在长文本中，术语的关系可能跨越多个句子或段落。请注意，跨句、跨段落的术语间可能存在直接或间接的关系。\n"
    "如果术语在多个句子中提到，并且它们之间有联系，考虑它们可能的关系。\n"

    "歧义处理：\n"
    "对于歧义术语，根据上下文来推测其具体含义。\n"
    "例如，“Apple”可能指公司，也可能指水果；“Bank”可能指金融机构或河岸。\n"

    "非常重要！！！！将你的输出格式必须转化为 JSON 列表。列表的每个元素包含一对术语和它们之间的关系，如下所示：\n"
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

"""
# 定义 SYS_PROMPT 和 USER_PROMPT

SYS_PROMPT = (
    "你是一个网络图制作者，从给定的上下文中提取术语及其关系。你的目标是生成一个清晰且简洁的知识图谱，并优先归一化相似的术语对。\n"
    "\n"
    "### 术语归一化\n"
    "1. 为术语选择通用、简洁的表达方式。例如，将‘Fraud detection system’和‘Fraud detection’统一为‘Fraud detection’。\n"
    "2. 优先选取术语在上下文中出现频率较高的名称作为统一的术语。\n"
    "3. 忽略细枝末节术语，仅保留对关系有明确意义的核心术语。例如，将‘network protocol used in IoT devices’简化为‘IoT network protocol’。\n"
    "\n"
    "### 关系归一化\n"
    "1. 对语义相似的关系用单一术语表示，例如：\n"
    "   ‘is part of’和‘belongs to’统一为‘part of’。\n"
    "   ‘is applied in’和‘used in’统一为‘applied in’。\n"
    "2. 删除冗余的弱关系，例如‘related to’仅在关系无法分类时使用。\n"
    "\n"
    "### 优先提取规则\n"
    "1. 术语提取优先顺序：\n"
    "   a. 核心主题术语，例如文档的标题、段落主语。\n"
    "   b. 出现频率最高的术语。\n"
    "   c. 明确标注的实体或概念。\n"
    "2. 忽略频率低且未明确标注为核心的术语。\n"
    "\n"
    "### 跨段关系\n"
    "1. 如果两个术语在不同段落中出现，且语义相同或高度相似，请将其视为同一术语。\n"
    "2. 在跨段落提取关系时，优先检测主题术语间的主要关系。\n"
    "\n"
    "### 歧义处理\n"
    "对于歧义术语，根据上下文来推测其具体含义。例如，“Apple”可能指公司，也可能指水果；“Bank”可能指金融机构或河岸。\n"
    "\n"
    "非常重要！！！！你的输出格式必须必须必须是 JSON 列表。列表的每个元素包含一对术语和它们之间的关系，如下所示：\n"
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

def process_chunk(client, chunk):
    """
    对每个文本块调用模型，返回 JSON 格式的关系
    """
    formatted_user_prompt = USER_PROMPT.format(input=chunk)
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
    try:
        # 解析响应为 JSON 格式
        return json.loads(response.message['content'])
    except json.JSONDecodeError:
        print("解析模型响应失败，跳过该块。")
        return []

def read_chunks_from_file(file_path):
    """
    从文件读取文本块
    """
    with open(file_path, "r", encoding="utf-8") as f:
        chunks = f.read().split("### Chunk")
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def chunks_to_dataframe(chunks_results):
    """
    将所有块的结果合并为一个 Pandas DataFrame
    """
    all_relations = []
    for chunk_result in chunks_results:
        all_relations.extend(chunk_result)
    df = pd.DataFrame(all_relations)
    # 移除自环
    df = df[df['node_1'] != df['node_2']]
    return df

def process_text(input_chunks_file):
    # 从文件读取所有文本块
    chunks = read_chunks_from_file(input_chunks_file)
    print(f"读取到 {len(chunks)} 个文本块。")

    # 逐块处理文本块并收集结果
    chunks_results = []
    for i, chunk in enumerate(chunks):
        print(f"正在处理第 {i + 1}/{len(chunks)} 块...")
        chunk_result = process_chunk(client, chunk)
        chunks_results.append(chunk_result)

    # 将结果合并为 Pandas DataFrame
    final_df = chunks_to_dataframe(chunks_results)
    return final_df

def save_relations_to_csv(df, output_csv):
    # 保存结果为 CSV 文件, 赋予覆盖写入权限
    df.to_csv(output_csv, index=False, encoding="utf-8", mode='w')
    print(f"处理完成，结果已保存到 {output_csv}")

def main(input_chunks_file, output_csv):
    # 调用处理函数
    final_df = process_text(input_chunks_file)
    save_relations_to_csv(final_df, output_csv)

"""
if __name__ == "__main__":
    # 获取命令行参数作为输入文本
    if len(sys.argv) > 2:
        input_chunks_file = sys.argv[1]
        output_csv = sys.argv[2]
    else:
        print("错误: 未提供输入参数。")
        sys.exit(1)

    # 调用处理函数
    main(input_chunks_file, output_csv)

"""