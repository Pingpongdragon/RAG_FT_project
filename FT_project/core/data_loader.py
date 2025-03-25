from config import settings
from langchain.schema import Document
from datasets import load_dataset
from pathlib import Path
from typing import List
from config.logger_config import configure_console_logger

logger = configure_console_logger(__name__)

def _load_hf_dataset() -> List[Document]:
    """根据配置加载数据集"""
    
    # 从配置读取参数
    cfg = settings.KNOWLEDGE_DATASET_CONFIG
    logger.info(f"🚀 正在加载数据集 {cfg['dataset_name']}")
    
    try:
        # 自动检测本地缓存路径
        cache_dir = Path(settings.DATA_CACHE_DIR) 
        raw_data = load_dataset(
            cfg['dataset_name'], 
            cfg['config_name'],
            split=cfg['split'],
            cache_dir=str(cache_dir)
        )
    except Exception as e:
        logger.error(f"🔥 数据集加载失败: {str(e)}")
        return []

    # 有效性校验（动态检查字段）
    required_columns = cfg['text_columns'] + [cfg['id_column']]
    valid_items = [
        item for item in raw_data 
        if all(k in item for k in required_columns)
    ]
    invalid_count = len(raw_data) - len(valid_items)
    
    logger.info(f"📊 加载完成: 有效 {len(valid_items)}条, 跳过无效 {invalid_count}条")
    
    # 组合多个文本字段
    return [
        Document(
            page_content= "\n".join(str(item[col]) for col in cfg['text_columns']),
            metadata={
                "doc_id": item[cfg['id_column']],
                "source": cfg['dataset_name']
            }
        ) for item in valid_items
    ]