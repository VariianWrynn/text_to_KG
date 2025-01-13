import pandas as pd

def postprocess_csv_file(input_file, output_file):
    # 读取 CSV 文件
    df = pd.read_csv(input_file)
    
    # 检查输入数据的结构
    if not {'node_1', 'node_2', 'edge'}.issubset(df.columns):
        raise ValueError("输入文件必须包含 'node_1', 'node_2', 'edge' 列。")

    # 将所有 edge 值统一为 "contextual proximity"
    df['edge'] = "contextual proximity"

    # 按 (node1, node2) 组合统计频次
    df['count'] = df.groupby(['node_1', 'node_2'])['node_2'].transform('count')

    # 删除重复行，保留每个 (node1, node2, edge) 的唯一记录
    df = df.drop_duplicates(subset=['node_1', 'node_2', 'edge'])

    # 保存结果到新的 CSV 文件
    df.to_csv(output_file, index=False)
    print(f"处理完成，结果已保存到 {output_file}")

"""
if __name__ == "__main__":
    # 输入和输出文件路径
    input_csv = "output_relations.csv"  # 替换为你的输入文件路径
    output_csv = "contextual_relations.csv"  # 替换为你的输出文件路径

    # 调用主函数处理 CSV 文件
    postprocess_csv_file(input_csv, output_csv)
"""
