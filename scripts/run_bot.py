#!/usr/bin/env python3
"""
Скрипт для запуска Telegram бота LegalGuardian.
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Добавляем родительскую директорию в путь для импортов
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Импорт функции запуска бота
from telegram_bot.bot import run_bot, initialize_bot

def main():
    """
    Основная функция для запуска бота
    """
    # Загрузка переменных окружения
    load_dotenv()
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Запуск юридического бота LegalGuardian")
    parser.add_argument("--telegram-token", type=str, default=os.environ.get("TELEGRAM_TOKEN"),
                        help="Токен для Telegram бота")
    parser.add_argument("--huggingface-token", type=str, default=os.environ.get("HUGGINGFACE_TOKEN"),
                        help="Токен для доступа к моделям Hugging Face")
    parser.add_argument("--index-path", type=str, default=os.environ.get("INDEX_PATH", "data/legal_index.faiss"),
                        help="Путь к индексу FAISS")
    parser.add_argument("--chunks-data-path", type=str, default=os.environ.get("CHUNKS_DATA_PATH", "data/chunks_references.pkl"),
                        help="Путь к файлу с чанками и ссылками")
    parser.add_argument("--model", type=str, default=os.environ.get("LLM_MODEL", "google/gemma-3-4b-it"),
                        help="Название модели для генерации ответов")
    
    args = parser.parse_args()
    
    # Проверка наличия обязательных аргументов
    if not args.telegram_token:
        print("Ошибка: Не указан токен Telegram бота. "
              "Укажите его через аргумент --telegram-token или переменную окружения TELEGRAM_TOKEN.")
        return 1
    
    # Проверка наличия индекса и данных чанков
    if not os.path.exists(args.index_path):
        print(f"Ошибка: Файл индекса не найден по пути {args.index_path}. "
              "Убедитесь, что индекс создан или укажите правильный путь через аргумент --index-path.")
        return 1
    
    if not os.path.exists(args.chunks_data_path):
        print(f"Ошибка: Файл с чанками не найден по пути {args.chunks_data_path}. "
              "Убедитесь, что данные чанков созданы или укажите правильный путь через аргумент --chunks-data-path.")
        return 1
    
    # Запуск бота
    try:
        print(f"Запуск бота LegalGuardian с моделью {args.model}...")
        
        # Инициализация и запуск бота
        bot = initialize_bot(
            telegram_token=args.telegram_token,
            index_path=args.index_path,
            chunks_data_path=args.chunks_data_path,
            llm_model_name=args.model,
            huggingface_token=args.huggingface_token
        )
        
        # Запуск бота
        bot.run()
        
        return 0
        
    except KeyboardInterrupt:
        print("\nРабота бота остановлена пользователем.")
        return 0
        
    except Exception as e:
        print(f"Ошибка при запуске бота: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 