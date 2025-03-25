from loguru import logger
import os


def configure_console_logger(name: str = __name__):
    """配置并返回用于控制台输出的日志记录器"""
    # 创建一个独立的日志记录器实例
    console_logger = logger.bind(name=name)
    # 移除默认的处理器（如果有）
    console_logger.remove()
    # 添加新的处理器，将日志输出到控制台
    console_logger.add(
        sink=lambda msg: print(msg, end=""),
        format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {module}:{line} - {message}",
        level="INFO"
    )
    return console_logger


def configure_file_logger(log_file: str, level: str = "INFO"):
    """为特定文件创建独立的日志记录器"""
    # 创建一个独立的日志记录器实例
    file_logger = logger.bind(file=log_file)
    
    # 先移除所有处理器，确保干净的状态
    file_logger.remove()
    
    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 添加文件处理器
    file_logger.add(
        sink=log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {module}:{line} - {message}",
        level=level,
        rotation="10 MB",
        retention="7 days"
    )
    return file_logger