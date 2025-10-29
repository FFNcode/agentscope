# -*- coding: utf-8 -*-
"""
Markdown 文件向量化处理示例

步骤：
1. 读取 md 文件
2. 提取 md 文件中的标题和内容
3. 将标题和内容转换为向量
4. 将向量存储到向量数据库中
5. 根据向量数据库中的向量搜索相似的标题和内容
6. 输出搜索结果
"""

import os
import re
import asyncio
from typing import List, Dict
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct

# 可以选择使用本地嵌入模型或 API 嵌入模型
try:
    from sentence_transformers import SentenceTransformer
    USE_LOCAL_EMBEDDING = True
except ImportError:
    USE_LOCAL_EMBEDDING = False
    print("未安装 sentence-transformers，将尝试使用 OpenAI 嵌入模型")
    try:
        from openai import OpenAI
        OPENAI_CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception:
        OPENAI_CLIENT = None


class MDProcessor:
    def __init__(
        self,
        file_path: str,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "md_documents",
        embedding_model_name: str = "all-MiniLM-L6-v2",  # sentence-transformers 模型
    ):
        """
        初始化 Markdown 处理器
        
        Args:
            file_path: Markdown 文件路径
            qdrant_url: Qdrant 服务器地址
            collection_name: 集合名称
            embedding_model_name: 嵌入模型名称
        """
        self.file_path = file_path
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        
        # 初始化嵌入模型
        if USE_LOCAL_EMBEDDING:
            print(f"加载本地嵌入模型: {embedding_model_name}")
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
        else:
            self.embedding_model = None
            # 如果使用 OpenAI，默认维度是 1536 (text-embedding-ada-002) 或 1536 (text-embedding-3-small)
            self.vector_size = 1536
        
        self.sections: List[Dict[str, str]] = []

    def read_md_file(self) -> str:
        """读取 Markdown 文件内容"""
        print(f"正在读取文件: {self.file_path}")
        with open(self.file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        print(f"文件读取完成，共 {len(content)} 字符")
        return content

    def extract_title_and_content(self, content: str) -> List[Dict[str, str]]:
        """
        提取 Markdown 文件中的标题和内容
        
        Args:
            content: Markdown 文件内容
            
        Returns:
            包含标题和内容的字典列表
        """
        print("\n正在提取标题和内容...")
        sections = []
        
        # 使用正则表达式匹配标题（# 开头的行）
        # 匹配模式：## 标题 或 ### 标题等
        lines = content.split('\n')
        current_title = "Introduction"  # 默认标题
        current_content = []
        
        for line in lines:
            # 检查是否是标题
            title_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if title_match:
                # 保存之前的章节
                if current_content:
                    content_text = '\n'.join(current_content)
                    sections.append({
                        'title': current_title,
                        'content': content_text.strip(),
                        'text': f"{current_title}\n{content_text}".strip()
                    })
                
                # 更新当前标题和内容
                current_title = title_match.group(2).strip()
                current_content = []
            else:
                # 添加到当前内容
                if line.strip():  # 忽略空行
                    current_content.append(line)
        
        # 添加最后一个章节
        if current_content:
            content_text = '\n'.join(current_content)
            sections.append({
                'title': current_title,
                'content': content_text.strip(),
                'text': f"{current_title}\n{content_text}".strip()
            })
        
        self.sections = sections
        print(f"提取完成，共找到 {len(sections)} 个章节")
        
        for i, section in enumerate(sections, 1):
            print(f"  章节 {i}: {section['title'][:50]}...")
        
        return sections

    async def convert_to_vector(self, texts: List[str]) -> List[List[float]]:
        """
        将文本转换为向量
        
        Args:
            texts: 文本列表
            
        Returns:
            向量列表
        """
        print(f"\n正在将 {len(texts)} 个文本转换为向量...")
        
        if USE_LOCAL_EMBEDDING and self.embedding_model:
            # 使用本地模型
            vectors = self.embedding_model.encode(texts, show_progress_bar=True)
            vectors = vectors.tolist() if hasattr(vectors, 'tolist') else vectors
        elif OPENAI_CLIENT:
            # 使用 OpenAI API
            response = OPENAI_CLIENT.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            vectors = [item.embedding for item in response.data]
            self.vector_size = len(vectors[0]) if vectors else 1536
        else:
            error_msg = (
                "未找到可用的嵌入模型。请安装 sentence-transformers "
                "或设置 OPENAI_API_KEY 环境变量"
            )
            raise ValueError(error_msg)
        
        print(f"向量转换完成，向量维度: {self.vector_size}")
        return vectors

    def ensure_collection(self):
        """确保集合存在且配置正确"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            # 检查向量维度是否匹配
            config = collection_info.config.params.vectors
            if hasattr(config, 'size') and config.size != self.vector_size:
                print(f"向量维度不匹配，删除旧集合并重新创建...")
                self.client.delete_collection(self.collection_name)
                self.create_collection()
            else:
                print(f"集合 '{self.collection_name}' 已存在")
        except Exception:
            print(f"集合 '{self.collection_name}' 不存在，正在创建...")
            self.create_collection()

    def create_collection(self):
        """创建集合"""
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        print(f"集合 '{self.collection_name}' 创建成功 (维度: {self.vector_size})")

    async def store_to_vector_database(self) -> None:
        """将向量存储到向量数据库"""
        if not self.sections:
            print("没有可存储的数据，请先提取标题和内容")
            return
        
        print("\n正在存储向量到数据库...")
        
        # 确保集合存在
        self.ensure_collection()
        
        # 提取文本用于向量化
        texts = [section['text'] for section in self.sections]
        
        # 转换为向量
        vectors = await self.convert_to_vector(texts)
        
        # 创建点并存储
        points = [
            PointStruct(
                id=i,
                vector=vector,
                payload={
                    'title': section['title'],
                    'content': section['content'],
                    'text': section['text'],
                    'file_path': self.file_path,
                }
            )
            for i, (section, vector) in enumerate(zip(self.sections, vectors))
        ]
        
        # 存储到 Qdrant
        operation_info = self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=points,
        )
        
        print(f"存储完成: {len(points)} 个文档已添加到数据库")
        print(f"操作信息: {operation_info}")

    async def search_similar_title_and_content(
        self,
        query: str,
        limit: int = 3,
    ) -> List[Dict]:
        """
        搜索相似的标题和内容
        
        Args:
            query: 查询文本
            limit: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        print(f"\n正在搜索: '{query}'")
        
        # 将查询文本转换为向量
        query_vectors = await self.convert_to_vector([query])
        query_vector = query_vectors[0]
        
        # 在 Qdrant 中搜索
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        
        results = []
        for point in search_result.points:
            results.append({
                'id': point.id,
                'score': point.score,
                'title': point.payload.get('title', ''),
                'content': point.payload.get('content', ''),
                'text': point.payload.get('text', ''),
            })
        
        return results

    def output_search_result(self, results: List[Dict]) -> None:
        """输出搜索结果"""
        print(f"\n找到 {len(results)} 个相关结果:\n")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\n结果 {i}:")
            print(f"  相似度得分: {result['score']:.4f}")
            print(f"  标题: {result['title']}")
            print(f"  内容预览: {result['content'][:200]}...")
            print("-" * 80)

    async def process(self) -> None:
        """执行完整的处理流程"""
        # Step 1: 读取文件
        content = self.read_md_file()
        
        # Step 2: 提取标题和内容
        self.extract_title_and_content(content)
        
        # Step 3-4: 转换为向量并存储到数据库
        await self.store_to_vector_database()
        
        # Step 5-6: 演示搜索功能
        print("\n" + "=" * 80)
        print("开始测试搜索功能")
        print("=" * 80)
        
        # 示例搜索查询
        test_queries = [
            "快速开始",
            "向量数据库",
            "如何安装",
            "搜索功能",
        ]
        
        for query in test_queries:
            results = await self.search_similar_title_and_content(query, limit=2)
            self.output_search_result(results)


async def main():
    """主函数"""
    # 使用当前目录下的 qdrant.md 文件
    file_path = os.path.join(
        os.path.dirname(__file__),
        "qdrant.md"
    )
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    # 创建处理器并执行
    processor = MDProcessor(file_path=file_path)
    await processor.process()
    
    # 交互式搜索
    print("\n" + "=" * 80)
    print("进入交互式搜索模式（输入 'exit' 退出）")
    print("=" * 80)
    
    while True:
        query = input("\n请输入搜索查询: ").strip()
        if not query or query.lower() == 'exit':
            break
        
        try:
            results = await processor.search_similar_title_and_content(query, limit=3)
            processor.output_search_result(results)
        except Exception as e:
            print(f"搜索出错: {e}")


if __name__ == "__main__":
    asyncio.run(main())
