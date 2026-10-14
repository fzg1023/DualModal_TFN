"""
配置日志器，输出到指定文件和控制台；统一日志格式（月 - 日 时：分: 秒 + 日志消息）；控制日志粒度（通过INFO级别过滤 DEBUG 级别的冗余日志，仅保留关键信息）
"""
import logging

def setlogger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s %(message)s", "%m-%d %H:%M:%S")

    fileHandler = logging.FileHandler(path)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
