# Qdrant 快速入门

本指南基于 [Qdrant 官方文档](https://qdrant.tech/documentation/)，介绍如何使用 Qdrant 向量数据库进行向量存储和语义搜索。

## 什么是 Qdrant？

Qdrant 是一个 AI 原生的向量数据库和语义搜索引擎。您可以使用它从未结构化数据中提取有意义的信息。

## 快速开始

### 安装 Qdrant

#### 方式 1: Python 客户端（推荐用于开发）

```bash
pip install qdrant-client
```

#### 方式 2: 使用 Docker

**使用 Docker 命令运行：**

```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
```

**使用 Docker Compose（推荐，支持数据持久化）：**

创建 `docker-compose.yml` 文件：

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant
    ports:
      - "6333:6333"  # REST API 端口
      - "6334:6334"  # gRPC 端口
    volumes:
      # 映射数据目录到本地，实现数据持久化
      - ./qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

启动服务：

```bash
docker-compose up -d
```

停止服务：

```bash
docker-compose down
```

查看日志：

```bash
docker-compose logs -f qdrant
```

#### 方式 3: Qdrant Cloud（云端服务）

访问 [Qdrant Cloud](https://qdrant.to) 创建免费集群。

更多安装选项请参考 [Qdrant 安装文档](https://qdrant.tech/documentation/guides/installation/)。

### 基本使用示例

```python
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# 连接到 Qdrant
# 方式 1: 内存模式（仅用于测试）
client = QdrantClient(":memory:")

# 方式 2: 本地模式
# client = QdrantClient(path="./qdrant_data")

# 方式 3: 远程服务器
# client = QdrantClient(url="http://localhost:6333")
# client = QdrantClient(url="https://your-cluster.qdrant.io", api_key="your-api-key")

# 创建集合
client.create_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(size=128, distance=Distance.COSINE),
)

# 添加点（Points）
points = [
    PointStruct(
        id=1,
        vector=[0.1, 0.2, 0.3, ...],  # 128 维向量
        payload={"text": "文档内容", "category": "科技"}
    ),
    PointStruct(
        id=2,
        vector=[0.2, 0.3, 0.4, ...],
        payload={"text": "另一个文档", "category": "新闻"}
    ),
]
client.upsert(collection_name="my_collection", points=points)

# 搜索
hits = client.search(
    collection_name="my_collection",
    query_vector=[0.15, 0.25, 0.35, ...],  # 查询向量
    limit=5,  # 返回前 5 个结果
)

for hit in hits:
    print(f"ID: {hit.id}, Score: {hit.score}, Payload: {hit.payload}")
```

## 核心概念

### 集合（Collections）

集合是 Qdrant 中存储向量的容器，类似于传统数据库中的表。

```python
from qdrant_client.models import VectorParams, Distance

# 创建集合
client.create_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# 获取集合信息
collection_info = client.get_collection("my_collection")
print(collection_info)

# 列出所有集合
collections = client.get_collections()
for collection in collections.collections:
    print(collection.name)

# 删除集合
client.delete_collection("my_collection")
```

### 点（Points）

点是 Qdrant 中的基本数据单元，每个点包含：
- **ID**: 唯一标识符
- **Vector**: 向量数据
- **Payload**: 可选的元数据（键值对）

```python
from qdrant_client.models import PointStruct

# 创建点
point = PointStruct(
    id=1,
    vector=[0.1, 0.2, 0.3, ...],  # 向量
    payload={  # 元数据
        "name": "示例文档",
        "author": "张三",
        "tags": ["AI", "机器学习"],
        "year": 2024,
    }
)

# 添加单个点
client.upsert(collection_name="my_collection", points=[point])

# 批量添加点
points = [
    PointStruct(id=i, vector=[...], payload={...})
    for i in range(100)
]
client.upsert(collection_name="my_collection", points=points)
```

### 向量（Vectors）

向量是数值数组，通常由嵌入模型生成。向量的维度必须与创建集合时指定的维度一致。

```python
# 创建不同维度的集合
client.create_collection(
    collection_name="collection_384",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

client.create_collection(
    collection_name="collection_1536",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)
```

### 负载（Payload）

Payload 是点的元数据，用于存储额外的结构化信息。

```python
# 带有复杂 payload 的点
point = PointStruct(
    id=1,
    vector=[...],
    payload={
        "text": "文档内容",
        "metadata": {
            "author": "张三",
            "created_at": "2024-01-01",
        },
        "tags": ["AI", "NLP"],
        "score": 0.95,
    }
)

# 更新点的 payload
client.set_payload(
    collection_name="my_collection",
    payload={
        "updated_at": "2024-01-02",
    },
    points=[1],
)
```

## 距离度量

Qdrant 支持以下距离度量方法：

| 距离度量 | 说明 | 适用场景 |
|---------|------|---------|
| **COSINE** | 余弦相似度 | 文本嵌入（最常用） |
| **EUCLID** | 欧几里得距离（L2） | 空间数据、图像 |
| **DOT** | 点积 | 归一化向量的内积 |
| **MANHATTAN** | 曼哈顿距离（L1） | 特定机器学习场景 |

```python
from qdrant_client.models import Distance

# 使用余弦距离（推荐用于文本）
client.create_collection(
    collection_name="text_collection",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# 使用欧几里得距离
client.create_collection(
    collection_name="image_collection",
    vectors_config=VectorParams(size=512, distance=Distance.EUCLID),
)
```

## 搜索操作

### 基本搜索

```python
# 向量搜索
hits = client.search(
    collection_name="my_collection",
    query_vector=[0.1, 0.2, 0.3, ...],
    limit=10,
)

for hit in hits:
    print(f"ID: {hit.id}, Score: {hit.score}")
```

### 带过滤条件的搜索

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

# 按 payload 字段过滤
hits = client.search(
    collection_name="my_collection",
    query_vector=[0.1, 0.2, 0.3, ...],
    query_filter=Filter(
        must=[
            FieldCondition(key="category", match=MatchValue(value="科技"))
        ]
    ),
    limit=10,
)
```

### 批量搜索

```python
# 同时搜索多个查询向量
queries = [
    [0.1, 0.2, 0.3, ...],
    [0.4, 0.5, 0.6, ...],
]

results = client.search_batch(
    collection_name="my_collection",
    requests=[
        {
            "query": query,
            "limit": 5,
        }
        for query in queries
    ],
)
```

## 删除操作

```python
# 根据 ID 删除点
client.delete(
    collection_name="my_collection",
    points_selector=[1, 2, 3],
)

# 根据过滤条件删除
from qdrant_client.models import Filter, FieldCondition, MatchValue

client.delete(
    collection_name="my_collection",
    points_selector=Filter(
        must=[
            FieldCondition(key="category", match=MatchValue(value="旧分类"))
        ]
    ),
)
```

## 部署模式

### 1. 内存模式（测试）

```python
client = QdrantClient(":memory:")
```

- 数据仅存储在内存中
- 适合快速测试和开发
- 程序退出后数据丢失

### 2. 本地模式（持久化）

```python
client = QdrantClient(path="./qdrant_data")
```

- 数据存储在本地文件系统
- 适合小型应用和开发环境
- 数据持久化保存

### 3. 远程服务器模式（生产环境）

```python
# 本地服务器
client = QdrantClient(url="http://localhost:6333")

# 远程服务器（需要认证）
client = QdrantClient(
    url="https://your-cluster.qdrant.io",
    api_key="your-api-key",
)
```

- 独立运行的 Qdrant 服务
- 支持分布式部署
- 适合生产环境

## 异步 API

Qdrant 客户端支持异步操作，适合高并发场景：

```python
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct
import asyncio

async def main():
    client = AsyncQdrantClient(":memory:")
    
    # 创建集合
    await client.create_collection(
        collection_name="async_collection",
        vectors_config=VectorParams(size=128, distance=Distance.COSINE),
    )
    
    # 添加点
    points = [PointStruct(id=i, vector=[0.1]*128) for i in range(10)]
    await client.upsert(collection_name="async_collection", points=points)
    
    # 搜索
    hits = await client.search(
        collection_name="async_collection",
        query_vector=[0.1]*128,
        limit=5,
    )
    print(hits)

asyncio.run(main())
```

## 高级功能

### 混合搜索（Hybrid Search）

结合向量搜索和关键字搜索：

```python
from qdrant_client.models import Query, QueryVector, QueryText

# 混合查询
result = client.query_points(
    collection_name="my_collection",
    query=Query(
        query=[
            QueryVector(vector=[0.1, 0.2, ...], weight=0.7),
            QueryText(text="关键词", weight=0.3),
        ]
    ),
    limit=10,
)
```

更多信息请参考 [混合查询文档](https://qdrant.tech/documentation/concepts/search/#hybrid-queries)。

### 快照（Snapshots）

创建集合快照用于备份和恢复：

```python
# 创建快照
snapshot_info = client.create_snapshot("my_collection")
print(snapshot_info)

# 列表快照
snapshots = client.list_snapshots("my_collection")
```

### 索引优化

Qdrant 使用 HNSW 算法进行高效向量搜索：

```python
from qdrant_client.models import HnswConfigDiff, OptimizersConfigDiff

client.update_collection(
    collection_name="my_collection",
    optimizer_config=OptimizersConfigDiff(
        indexing_threshold=10000,  # 达到 10000 个点时建立索引
    ),
    hnsw_config=HnswConfigDiff(
        m=16,  # HNSW 参数
        ef_construct=100,
    ),
)
```

## 常见问题

**如何选择合适的向量维度？**

向量维度由您使用的嵌入模型决定，常见维度：
- OpenAI `text-embedding-ada-002`: 1536
- OpenAI `text-embedding-3-small`: 1536
- OpenAI `text-embedding-3-large`: 3072
- 大多数 BERT 模型: 768
- 自定义模型: 根据模型输出确定

**如何选择合适的距离度量？**

- **文本数据**: 推荐使用 COSINE（余弦距离）
- **图像数据**: 可以使用 EUCLID 或 COSINE
- **归一化向量**: 可以使用 DOT

**内存模式和本地模式有什么区别？**

- 内存模式（`:memory:`）: 数据不持久化，适合测试
- 本地模式（路径）: 数据持久化到磁盘，适合生产

**如何优化搜索性能？**

1. 使用合适的索引参数（HNSW）
2. 考虑使用量化（Quantization）减少内存占用
3. 使用过滤条件减少搜索空间
4. 对于大规模数据，考虑分布式部署

**如何迁移到 Qdrant？**

Qdrant 提供 [迁移指南](https://qdrant.tech/documentation/tutorials/migration-to-qdrant/)，帮助从其他向量数据库迁移。

## 参考资源

- [Qdrant 官方文档](https://qdrant.tech/documentation/)
- [Qdrant 教程](https://qdrant.tech/documentation/tutorials/)
- [Qdrant API 参考](https://qdrant.github.io/qdrant-client/)
- [Qdrant Python 客户端](https://github.com/qdrant/qdrant-client)
- [Qdrant Cloud](https://qdrant.to)
