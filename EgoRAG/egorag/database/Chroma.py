from abc import ABC, abstractmethod
from typing import Any, List, Literal, Optional
import numpy as np
import time
import os
import torch
try:
    import chromadb
    from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
    from chromadb.utils.data_loaders import ImageLoader
except ImportError:
    raise ImportError("Chroma is not installed. Please install it with 'pip install chromadb'.")



class Chroma(ABC):
    def __init__(self, name='EgoRAG', method='text', db_dir='./chroma_db'):
        self.name = name
        self.method = method
        self.db_dir = db_dir
        self.db_path=os.path.join(self.db_dir,self.name)
        # Initialize client
        self._client = chromadb.PersistentClient(path=self.db_path)
        self.embedding_function = OpenCLIPEmbeddingFunction()
        
        # Get or create collection
        try:
            self._collection = self._client.get_collection(
                name=self.name, 
                embedding_function=self.embedding_function,
                data_loader=ImageLoader()
            )
            print(f'Find target collection {self.name}')
        except:
            self._collection = self._client.create_collection(
                name=self.name,
                embedding_function=self.embedding_function, 
                data_loader=ImageLoader()
            )
            print(f'Create new collection {self.name}')
    
    @property
    def client(self):
        return self._client

    @property
    def collection(self):
        return self._collection

    def get_content_by_id(self, id: str) -> Optional[dict]:
        try:
            result = self._collection.get(ids=[id])
            if result and len(result['documents']) > 0:
                return result['documents'][0]  # Assuming result includes 'documents'
            else:
                print(f'No content found for ID: {id}')
                return None
        except Exception as e:
            print(f'Error retrieving content by ID: {e}')
            return None
    def get_caption(self,id:str,n_result=0):
        def generate_id_range(id: str, n: int):
            try:
                # 解析ID中的索引部分 (例如: DAY3_10000000_10003000_1 -> 1)
                parts = id.split('_')
                base_id = '_'.join(parts[:-1])  # 提取前面的部分，得到 "DAY3_10000000_10003000"
                index = int(parts[-1])  # 提取最后的部分，得到索引
                
                # 计算前后n句的范围，确保最小值为0
                start_index = max(0, index - n)
                end_index = index + n
                
                # 生成包含前后n句的ID列表
                ids_range = [f"{base_id}_{i}" for i in range(start_index, end_index + 1)]
                
                return ids_range
            except Exception as e:
                print(f'Error generating ID range: {e}')
                return None
        ids=generate_id_range(id=id,n=n_result)
        
        try:
            result = self._collection.get(ids=ids)
            if result and len(result['documents']) > 0:
                return {
                    "item": {
                        "ids": [id],
                        "documents": result['documents'],
                        "metadatas": [result['metadatas'][0]]
                        },
                    "documents": result['documents']
                }
            else:
                print(f'No content found for ID: {id}')
                return None
        except Exception as e:
            print(f'Error retrieving content by ID: {e}')
            return None
   
    def view_database(self, n: int = None, show_embeddings: bool = False):
        try:
            all_data = self._collection.get(include=['embeddings','documents','metadatas'])
            print(f"Number of entries in the database: {len(all_data['ids'])}")
            documents_to_view = all_data['documents'][:n] if n is not None else all_data['documents']
            for idx, document in enumerate(documents_to_view):
                metadata = all_data['metadatas'][idx] if 'metadatas' in all_data else 'No metadata'
                metadata_types = {key: type(value).__name__ for key, value in metadata.items()} if metadata != 'No metadata' else 'No metadata'
                output = f"ID: {all_data['ids'][idx]}, Content: {document}, Metadata: {metadata}, Metadata Types: {metadata_types}"
                if show_embeddings and 'embeddings' in all_data:
                    output += f", Embedding shape: {len(all_data['embeddings'][idx])}"
                print(output)
        except Exception as e:
            print(f'Error retrieving the whole dataset: {e}')

    def get_doc(self, n: int = None):
        all_docs = []
        
        all_data = self._collection.get()
        print(f"Number of entries in the database: {len(all_data['ids'])}")
        documents_to_view = all_data['documents'][:n] if n is not None else all_data['documents']
        
        for idx, document in enumerate(documents_to_view):
            try:
                
                metadata = all_data['metadatas'][idx] if 'metadatas' in all_data else 'No metadata'
                metadata_filtered = {key: metadata[key] for key in ['date', 'end_time', 'start_time']}
                all_docs.append({"Content": document, "Metadata": metadata_filtered})
            except Exception as e:
                print(f'Error processing entry {idx}: {e}')
                continue
                
        return all_docs
    
    def custom_query(
        self,
        query_texts: List[str],
        n_results: int = 3,
        where: dict = None,
        filter_first: bool = True
    ) -> dict:
        """
        自定义查询方法，支持从过滤后的数据中直接检索
        """
        try:
            if not filter_first or not where:
                return self._collection.query(
                    query_texts=query_texts,
                    n_results=n_results,
                    where=where
                )

            # 1. 获取过滤后的数据（明确包含embeddings）
            #start_time = time.time()
            filtered_data = self._collection.get(
                where=where,
                include=['embeddings', 'documents', 'metadatas']  # 明确指定要包含embeddings
            )
            #filter_time = time.time() - start_time
            #print(f"过滤数据耗时: {filter_time:.4f}秒")
           
            if not filtered_data['ids']:
                return {'ids': [], 'documents': [], 'metadatas': [], 'distances': []}

            # 2. 获取查询文本的嵌入向量
            #start_time = time.time()
            query_embeddings = self.embedding_function(query_texts)
            #query_embed_time = time.time() - start_time
            #print(f"查询文本编码耗时: {query_embed_time:.4f}秒")
          
            # 3. 获取过滤后数据的嵌入向量
            #start_time = time.time()
            if filtered_data['embeddings'] is None:
                filtered_embeddings = np.array([self.embedding_function([doc])[0] for doc in filtered_data['documents']])
            else:
                filtered_embeddings = np.array(filtered_data['embeddings'])
            #filter_embed_time = time.time() - start_time
            #print(f"过滤数据编码耗时: {filter_embed_time:.4f}秒")
        
            # 4. 计算相似度并获取结果
            #start_time = time.time()
            similarities = np.dot(query_embeddings, filtered_embeddings.T) / (
                np.linalg.norm(query_embeddings, axis=1)[:, np.newaxis] *
                np.linalg.norm(filtered_embeddings, axis=1)
            )

            n_results = min(n_results, len(filtered_data['ids']))
            top_indices = np.argsort(similarities[0])[-n_results:][::-1]

            results = {
                'ids': [filtered_data['ids'][i] for i in top_indices],
                'documents': [filtered_data['documents'][i] for i in top_indices],
                'metadatas': [filtered_data['metadatas'][i] for i in top_indices],
                'distances': [1 - similarities[0][i] for i in top_indices]
            }
            #similarity_time = time.time() - start_time
            #print(f"相似度计算耗时: {similarity_time:.4f}秒")
            #print(f"总耗时: {filter_time + query_embed_time + filter_embed_time + similarity_time:.4f}秒")

            return results

        except Exception as e:
            print(f'Error in custom query: {e}')
            return {'ids': [], 'documents': [], 'metadatas': [], 'distances': []}

    def clean_database(self):
        """
        清空当前数据库中的所有数据
        """
        try:
            self._collection.delete(self._collection.get()["ids"])
            print(f"All data in the collection {self.name} has been deleted.")
        except Exception as e:
            print(f'Error cleaning the database: {e}')
    
    def check_metadata_completeness(self, required_keys=None):
        """
        检查数据库中的每条数据是否都包含指定的metadata键
        Args:
            required_keys (list): 需要检查的metadata键列表，如果为None则只检查是否有metadata
        Returns:
            dict: 包含检查结果的字典
        """
        try:
            all_data = self._collection.get()
            total_entries = len(all_data['ids'])
            missing_metadata = []
            metadata_stats = {
                "entries_with_metadata": 0,
                "entries_without_metadata": 0,
                "metadata_key_frequency": {}
            }
            
            if total_entries == 0:
                return {"status": "empty", "message": "数据库中没有数据"}
            
            # 统计所有出现的metadata键
            all_keys = set()
            for metadata in all_data['metadatas']:
                if metadata != 'No metadata' and metadata is not None:
                    all_keys.update(metadata.keys())
            
            # 初始化键频率统计
            for key in all_keys:
                metadata_stats["metadata_key_frequency"][key] = 0
            
            for idx, metadata in enumerate(all_data['metadatas']):
                if metadata == 'No metadata' or metadata is None:
                    metadata_stats["entries_without_metadata"] += 1
                    missing_metadata.append({
                        "id": all_data['ids'][idx],
                        "issue": "完全缺失metadata"
                    })
                    continue
                
                metadata_stats["entries_with_metadata"] += 1
                # 统计每个键的出现频率
                for key in metadata:
                    metadata_stats["metadata_key_frequency"][key] += 1
                
                if required_keys:
                    missing_keys = [key for key in required_keys if key not in metadata]
                    if missing_keys:
                        missing_metadata.append({
                            "id": all_data['ids'][idx],
                            "issue": f"缺失以下键: {missing_keys}"
                        })
            
            # 计算每个键的覆盖率百分比
            for key in metadata_stats["metadata_key_frequency"]:
                frequency = metadata_stats["metadata_key_frequency"][key]
                coverage = (frequency / total_entries) * 100
                metadata_stats["metadata_key_frequency"][key] = {
                    "count": frequency,
                    "coverage_percentage": round(coverage, 2)
                }
            
            result = {
                "total_entries": total_entries,
                "entries_with_issues": len(missing_metadata),
                "all_complete": len(missing_metadata) == 0,
                "metadata_statistics": metadata_stats,
                "issues": missing_metadata if missing_metadata else None
            }
            
            return result
            
        except Exception as e:
            print(f'检查metadata时发生错误: {e}')
            return {"status": "error", "message": str(e)}


    def check_and_clean_empty_entries(self):
        """
        检查所有条目，删除documents或metadatas为空的条目。
        返回包含被删除ID列表、数量及状态信息的字典。
        """
        try:
            # 获取所有数据，包含documents和metadatas
            all_data = self._collection.get(include=['documents', 'metadatas'])
            ids = all_data['ids']
            documents = all_data['documents']
            metadatas = all_data['metadatas']

            ids_to_delete = []
            for idx in range(len(ids)):
                doc = documents[idx]
                meta = metadatas[idx] if idx < len(metadatas) else None

                # 检查document是否为空（None或空字符串）
                doc_empty = doc is None or (isinstance(doc, str) and doc.strip() == '')
                # 检查metadata是否为空（None或空字典）
                meta_empty = meta is None or meta == {}

                if doc_empty or meta_empty:
                    ids_to_delete.append(ids[idx])

            if ids_to_delete:
                self._collection.delete(ids=ids_to_delete)
                print(f"Deleted {len(ids_to_delete)} entries with empty documents or metadatas.")
                return {
                    "deleted_ids": ids_to_delete,
                    "count": len(ids_to_delete),
                    "message": f"Successfully deleted {len(ids_to_delete)} entries."
                }
            else:
                print("No empty entries found.")
                return {
                    "deleted_ids": [],
                    "count": 0,
                    "message": "All entries are valid. No deletions needed."
                }
        except Exception as e:
            print(f"Error checking and cleaning empty entries: {e}")
            return {
                "error": str(e),
                "message": "An error occurred during the cleaning process."
            }
if __name__ == '__main__':

    a = Chroma(name='EgoRAG')
