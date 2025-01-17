import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

    calculate_tfidf_similarity(output_file, "output_similarity_score.csv")

def calculate_tfidf_similarity(input_csv, output_similarity_csv):
    """
    基于术语对和关系描述计算 TF-IDF 相似度
    :param input_csv: 输入的术语对 CSV 文件路径
    :param output_similarity_csv: 输出的相似度结果 CSV 文件路径
    """
    # 读取 CSV 文件
    df = pd.read_csv(input_csv)
    
    # 检查必须的列是否存在
    if not {'node_1', 'node_2', 'edge'}.issubset(df.columns):
        raise ValueError("输入 CSV 文件必须包含 'node_1', 'node_2', 'edge' 列")

    # 合并术语对和关系描述为唯一标识符
    df['combined'] = df['node_1'] + " -> " + df['node_2'] + ": " + df['edge']

    # 提取关系描述作为语料
    corpus = df['combined']

    # 使用 TF-IDF 向量化关系描述
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # 计算余弦相似度矩阵
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # 构建相似性 DataFrame
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=df['combined'],
        columns=df['combined']
    )

    # 将相似性结果保存为 CSV 文件
    similarity_df.to_csv(output_similarity_csv, encoding='utf-8')
    print(f"TF-IDF 相似度矩阵已保存到 {output_similarity_csv}")

    return similarity_df


"""
if __name__ == "__main__":
    # 输入和输出文件路径
    input_csv = "output_relations.csv"  # 替换为你的输入文件路径
    output_csv = "contextual_relations.csv"  # 替换为你的输出文件路径

    # 调用主函数处理 CSV 文件
    postprocess_csv_file(input_csv, output_csv)
"""
