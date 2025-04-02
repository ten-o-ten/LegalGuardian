#!/usr/bin/env python3
"""
LegalGuardianBot - Запуск телеграм-бота с юридическими консультациями
Этот скрипт запускает телеграм-бота, который отвечает на юридические вопросы
на основе индексированной базы законодательства и модели Gemma-3.
"""

import os
import asyncio
import logging
from dotenv import load_dotenv
from huggingface_hub import login

from bot import LegalGuardianBot

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    """
    Основная функция запуска бота
    Загружает переменные окружения, авторизуется на Hugging Face
    и запускает телеграм-бота
    """
    # Загрузка переменных окружения из .env файла
    load_dotenv()
    
    # Получение необходимых токенов и путей из переменных окружения
    telegram_token = os.getenv("TELEGRAM_TOKEN")
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    index_path = os.getenv("INDEX_PATH", "data/legal_index.faiss")
    chunks_data_path = os.getenv("CHUNKS_DATA_PATH", "data/chunks_references.pkl")
    
    # Проверка наличия необходимых токенов
    if not telegram_token:
        logger.error("Токен Telegram бота не найден! Укажите TELEGRAM_TOKEN в .env файле")
        return
    
    if not huggingface_token:
        logger.error("Токен Hugging Face не найден! Укажите HUGGINGFACE_TOKEN в .env файле")
        return
    
    # Авторизация на Hugging Face
    try:
        login(token=huggingface_token)
        logger.info("Успешная авторизация на Hugging Face")
    except Exception as e:
        logger.error(f"Ошибка при авторизации на Hugging Face: {e}")
        return
    
    # Проверка наличия файлов индекса и чанков
    if not os.path.exists(index_path):
        logger.error(f"Файл индекса не найден: {index_path}")
        logger.info("Укажите правильный путь в .env файле (переменная INDEX_PATH)")
        return
    
    if not os.path.exists(chunks_data_path):
        logger.error(f"Файл чанков не найден: {chunks_data_path}")
        logger.info("Укажите правильный путь в .env файле (переменная CHUNKS_DATA_PATH)")
        return
    
    # Создание и запуск бота
    logger.info("Инициализация LegalGuardianBot...")
    
    try:
        # Создаем и запускаем бота асинхронно
        async def run_bot():
            bot = LegalGuardianBot(
                telegram_token=telegram_token,
                index_path=index_path,
                chunks_data_path=chunks_data_path,
                huggingface_token=huggingface_token,
                max_history=int(os.getenv("MAX_HISTORY", "8")),
                max_chunks=int(os.getenv("MAX_CHUNKS", "5")),
                max_tokens=int(os.getenv("MAX_TOKENS", "1024")),
                temperature=float(os.getenv("TEMPERATURE", "0.7")),
                top_p=float(os.getenv("TOP_P", "0.9"))
            )
            await bot.run()
        
        # Запуск асинхронной функции
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"Ошибка при запуске бота: {e}")

if __name__ == "__main__":
    main() 