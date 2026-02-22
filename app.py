import streamlit as st
#import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
#import re

# -------------------------- 页面基础配置 --------------------------
st.set_page_config(
    page_title="ISOM5580 Semantic Search",
    page_icon="🔍",
    layout="wide"
)


# -------------------------- 缓存函数（修复核心：不可哈希参数处理） --------------------------
@st.cache_resource(show_spinner="正在加载语义模型...")
def load_embedding_model():
    """加载语义嵌入模型（缓存，仅加载一次）"""
    try:
        # 轻量级模型，平衡速度和效果
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"模型加载失败：{str(e)}")
        st.stop()


@st.cache_data(show_spinner="正在处理文本并生成语义向量...")
def process_text_and_generate_embeddings(text, _model):  # 修复：model → _model（下划线跳过哈希）
    """处理文本（分块）并生成嵌入向量"""
    # 1. 文本分块：按段落分割，过滤空行
    chunks = [chunk.strip() for chunk in re.split(r'\n\s*\n', text) if chunk.strip()]

    # 如果段落太少，按句子分块（备用方案）
    if len(chunks) < 3:
        chunks = [sent.strip() for sent in re.split(r'(?<=[。！？；])', text) if sent.strip()]

    # 2. 生成每个文本块的嵌入向量（同步修改为 _model）
    embeddings = _model.encode(chunks, convert_to_numpy=True)

    return chunks, embeddings


# -------------------------- 核心搜索函数 --------------------------
def semantic_search(query, chunks, embeddings, model, top_k=5):
    """
    语义搜索核心逻辑
    :param query: 用户搜索查询
    :param chunks: 文本块列表
    :param embeddings: 文本块对应的嵌入向量
    :param model: 嵌入模型
    :param top_k: 返回最相似的top_k个结果
    :return: 排序后的结果（包含文本块、相似度）
    """
    # 生成查询的嵌入向量
    query_embedding = model.encode([query], convert_to_numpy=True)

    # 计算余弦相似度
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # 按相似度排序，获取top_k结果
    sorted_indices = similarities.argsort()[::-1][:top_k]
    results = [
        {
            "text": chunks[idx],
            "similarity": round(similarities[idx] * 100, 2)
        }
        for idx in sorted_indices
    ]

    return results


# -------------------------- 主界面逻辑 --------------------------
def main():
    st.title("🔍 语义搜索应用")
    st.divider()

    # 加载模型
    model = load_embedding_model()

    # 侧边栏：数据源设置
    with st.sidebar:
        st.subheader("数据源设置")
        # 选项1：上传文本文件
        uploaded_file = st.file_uploader(
            "上传文本文件（.txt）",
            type=["txt"],
            help="支持纯文本文件，编码建议UTF-8"
        )

        # 选项2：手动输入文本
        manual_text = st.text_area(
            "或手动输入文本内容",
            height=200,
            placeholder="请粘贴需要搜索的文本内容..."
        )

        # 搜索参数
        top_k = st.slider("返回结果数量", min_value=1, max_value=10, value=5)

    # 处理数据源
    text_source = None
    if uploaded_file is not None:
        # 读取上传的文件（兼容UTF-8/GBK编码）
        try:
            text_source = uploaded_file.read().decode("utf-8")
            st.success(f"✅ 成功加载文件：{uploaded_file.name}")
        except UnicodeDecodeError:
            text_source = uploaded_file.read().decode("gbk")
            st.success(f"✅ 成功加载文件（GBK编码）：{uploaded_file.name}")
        except Exception as e:
            st.error(f"文件读取失败：{str(e)}")
    elif manual_text:
        text_source = manual_text
        st.success("✅ 成功加载手动输入的文本")

    # 搜索区域
    st.subheader("开始语义搜索")
    query = st.text_input("输入搜索关键词/句子", placeholder="例如：人工智能的应用场景...")
    search_btn = st.button("搜索", type="primary", disabled=not (text_source and query))

    # 执行搜索并展示结果
    if search_btn and text_source and query:
        with st.spinner("正在进行语义匹配..."):
            # 处理文本并生成向量（传入model，函数内是 _model）
            chunks, embeddings = process_text_and_generate_embeddings(text_source, model)

            # 执行语义搜索
            results = semantic_search(query, chunks, embeddings, model, top_k)

            # 展示结果
            st.divider()
            st.subheader(f"🔎 搜索结果（Top {top_k}）")
            for idx, result in enumerate(results, 1):
                with st.expander(f"结果 {idx}（相似度：{result['similarity']}%）"):
                    st.write(result["text"])

    # 提示信息
    if not text_source:
        st.info("💡 请先在侧边栏上传文本文件或手动输入文本内容")
    elif text_source and not query:
        st.info("💡 请输入搜索关键词/句子后点击搜索按钮")


# -------------------------- 程序入口 --------------------------
if __name__ == "__main__":
    main()