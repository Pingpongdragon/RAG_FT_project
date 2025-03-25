from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# -------------------------
# 模型配置
# ------------------------- 
MODEL_NAME = "Meta-Llama-3.1-8B-Instruct"  # 模型名称
MODEL_DIR = str(BASE_DIR.parent / f"llm/LLM-Research/{MODEL_NAME}")  # 模型目录
ADAPTER_DIR = None  # 适配器目录
OUTPUT_DIR = str(BASE_DIR.parent / f"adaptor/output/{MODEL_NAME}")  # 适配器输出目录


# -------------------------
# 数据集配置
# -------------------------
DATA_DIR = ""

# -------------------------
# 训练参数配置
# -------------------------
lora_rank = 8
lora_alpha = 32
learning_rate = 1e-4
num_train_epochs = 3
per_device_train_batch_size = 1
save_steps = 50
save_total_limit = 5
logging_steps = 50




