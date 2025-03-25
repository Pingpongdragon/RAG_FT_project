from ..config import settings
from swift.llm import sft_main, TrainArguments

def train_model(data_dir):
    # 准备数据集，这里简单示例
    train_dataset = data_dir

    # 训练参数
    train_args = TrainArguments(
        dataset=train_dataset,
        resume_from_checkpoint = settings.ADAPTER_DIR,
        output_dir = settings.OUTPUT_DIR,
        model = settings.MODEL_DIR,
        lora_rank=settings.lora_rank,
        lora_alpha=settings.lora_alpha,
        learning_rate=settings.learning_rate,
        num_train_epochs=settings.num_train_epochs,
        per_device_train_batch_size=settings.per_device_train_batch_size,
        save_steps=settings.save_steps,
        save_total_limit=settings.save_total_limit,
        logging_steps=settings.logging_steps
    )

    return sft_main(train_args)


if __name__ == '__main__':
    train_model(data_dir='../../medicine-FT')