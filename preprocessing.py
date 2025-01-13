import os
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docx import Document
import pdfplumber

def read_word_file(file_path):
    """
    读取 Word 文件内容
    """
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        if para.text.strip():
            text += para.text + "\n"
    return text

def read_pdf_file(file_path):
    """
    读取 PDF 文件内容
    """
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            if page.extract_text():  # 确保页面有内容
                text += page.extract_text() + "\n"
    return text

def split_text(text, chunk_size=1000, chunk_overlap=100):
    """
    使用 LangChain 的 RecursiveCharacterTextSplitter 将文本切割成块
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "]
    )
    chunks = text_splitter.split_text(text)
    
    # 调试信息
    print(f"切分后的块数：{len(chunks)}")
    return chunks

def save_chunks_to_file(chunks, output_file):
    """
    将切分后的文本块保存到文件
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"### Chunk {i + 1} ###\n")
            f.write(chunk.strip())  # 去掉多余的空格或换行
            f.write("\n\n")

def preprocess_document(file_path, output_file, chunk_size=1000, chunk_overlap=100):
    # 判断文件类型
    if file_path.endswith(".docx"):
        text = read_word_file(file_path)
    elif file_path.endswith(".pdf"):
        text = read_pdf_file(file_path)
    else:
        raise ValueError("仅支持 Word (.docx) 和 PDF 文件 (.pdf)")

    # 切割文本
    chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"文档已切分为 {len(chunks)} 块。")

    # 保存切割后的块到文件
    save_chunks_to_file(chunks, output_file)
    print(f"切分后的文本已保存到文件：{output_file}")

"""
if __name__ == "__main__":
    # 检查是否提供了命令行参数
    if len(sys.argv) < 2:
        print("错误: 未提供输入文件路径。")
        sys.exit(1)

    # 获取命令行参数作为输入文件路径
    input_file = sys.argv[1]
    output_file = "output_chunks.txt"  # 替换为你的输出文件路径

    # 调用主函数
    preprocess_document(input_file, output_file, chunk_size=300, chunk_overlap=30)

"""