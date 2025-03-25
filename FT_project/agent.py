from contextlib import contextmanager
from .config import settings
from .core.train import train_model
from .config.logger_config import configure_console_logger,configure_file_logger
from typing import Dict
import random

logger = configure_console_logger(__name__)

class FTAgent:
    def __init__(self):
        self.default_params = {
            'lora_rank': settings.lora_rank,
            'lora_alpha': settings.lora_alpha,
            'learning_rate': settings.learning_rate,
            'num_train_epochs': settings.num_train_epochs,
            'per_device_train_batch_size': settings.per_device_train_batch_size,
        }

    def update_settings(self, params: Dict):
        """覆盖配置参数"""
        # 保存原始值并设置新值
        for key, value in params.items():
            if hasattr(settings, key):
                logger.info(f"\n🎛️  更新参数 {key} = {value}")
                setattr(settings, key, value)
            else:
                logger.warning(f"Ignored invalid parameter: {key}")

    def _random_adjust_params(self) -> Dict:
        """随机参数调整策略"""
        params = {}
        for param, default_value in self.default_params.items():
            # 为每个参数生成一个随机的调整因子，这里简单假设在0.95到1.05倍之间
            adjustment_factor = random.uniform(0.98, 1.02)
            new_value = default_value * adjustment_factor
            # 如果默认值是整数，则将调整后的值取整
            params[param] = int(new_value + 0.5) if isinstance(default_value, int) else new_value
        return params

    def train(self, data_dir: str) -> str:
        """训练入口函数"""
        # 1. 随机生成参数
        adjusted_params = self._random_adjust_params()
        logger.info(f"🎛️  调整后的参数配置: \n{adjusted_params}")

        # 2. 使用调整后的参数运行问答流程
        self.update_settings(adjusted_params)
        logger.info("🚀 启动微调流程...")
        return train_model(data_dir)


# 使用示例
if __name__ == "__main__":
    ft_agent = FTAgent()
    data_dir = input("请输入微调路径：")
    print(f"return : {ft_agent.train(data_dir)}")
