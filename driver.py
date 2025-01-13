#this function is the main driver.
# run preprocessing.py with parameter: sample.docx or sample.pdf

from preprocessing import preprocess_document
from zephyr import main as zephyr_main
from postprocessing import postprocess_csv_file

# 输入和输出文件路径
input_file = "sample.docx"  # 替换为你的输入文件路径
output_chunks_file = "output_chunks.txt"
output_relations_file = "output_relations.csv"
output_contextual_relations_file = "contextual_relations.csv"

def process_document(input_file, output_chunks_file):
    preprocess_document(input_file, output_chunks_file)

def zepry_main(input_chunks_file, output_csv):
    zephyr_main(input_chunks_file, output_csv)

def postprocess_csv(input_csv, output_csv):
    postprocess_csv_file(input_csv, output_csv)

if __name__ == "__main__":
    # 预处理文档
    preprocess_document(input_file, output_chunks_file)
    # 调用主函数处理文本块
    zepry_main(output_chunks_file, output_relations_file)
    # 后处理 CSV 文件
    postprocess_csv(output_relations_file, output_contextual_relations_file)