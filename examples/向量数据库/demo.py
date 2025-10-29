# -*- coding: utf-8 -*-
"""Qdrant 基本操作示例"""

from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")
collection_name = "test_collection"

# 向量配置
VECTOR_SIZE = 4
VECTOR_DISTANCE = models.Distance.COSINE


def ensure_collection(recreate: bool = False):
    """
    确保集合存在且配置正确
    
    Args:
        recreate: 如果为 True，则删除现有集合并重新创建
    """
    client.delete_collection(collection_name)
    create_collection()


def create_collection():
    """创建集合"""
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=VECTOR_DISTANCE,
        ),
        strict_mode_config=models.StrictModeConfig(
            enabled=True,
            unindexed_filtering_retrieve=False
        ),
    )
    print(f"集合 '{collection_name}' 创建成功")


def main():
    # 确保集合存在且配置正确
    # 如果需要强制删除重建，设置 recreate=True
    ensure_collection(recreate=False)

    # 添加一些测试数据
    print("\n添加测试数据...")
    operation_info = client.upsert(
        collection_name=collection_name,
        wait=True,
        points=[
            models.PointStruct(
                id=1,
                vector=[0.05, 0.61, 0.76, 0.74],
                payload={"city": "Berlin"}
            ),
            models.PointStruct(
                id=2,
                vector=[0.19, 0.81, 0.75, 0.11],
                payload={"city": "London"}
            ),
            models.PointStruct(
                id=3,
                vector=[0.36, 0.55, 0.47, 0.94],
                payload={"city": "Moscow"}
            ),
            models.PointStruct(
                id=4,
                vector=[0.18, 0.01, 0.85, 0.80],
                payload={"city": "New York"}
            ),
            models.PointStruct(
                id=5,
                vector=[0.24, 0.18, 0.22, 0.44],
                payload={"city": "Beijing"}
            ),
            models.PointStruct(
                id=6,
                vector=[0.35, 0.08, 0.11, 0.44],
                payload={"city": "Mumbai"}
            ),
        ],
    )
    print(f"添加完成: {operation_info}")

    # 搜索相似的向量
    print("\n搜索相似向量...")
    search_result = client.query_points(
        collection_name=collection_name,
        query=[0.2, 0.1, 0.9, 0.7],
        with_payload=True,  # 返回 payload
        limit=3,
    )

    print(f"\n找到 {len(search_result.points)} 个结果:")
    for point in search_result.points:
        print(f"  ID: {point.id}, Score: {point.score:.4f}, "
              f"Payload: {point.payload}")

if __name__ == "__main__":
    main()