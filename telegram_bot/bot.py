#!/usr/bin/env python3
"""
Основной модуль Telegram бота для ответов на юридические вопросы.
Обрабатывает взаимодействие с пользователями, запускает поиск релевантной
информации и генерирует ответы на основе найденных данных.
"""

import os
import logging
import asyncio
from typing import Dict
from datetime import datetime
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from huggingface_hub import login

# Импорт наших модулей
from telegram_bot.retriever import LegalRetriever
from telegram_bot.generator import LegalAnswerGenerator, ConversationMemory

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class LegalGuardianBot:
    """
    Telegram бот для ответов на юридические вопросы
    на основе базы знаний с российскими законами.
    """
    
    def __init__(
        self,
        telegram_token: str,
        index_path: str,
        chunks_data_path: str,
        llm_model_name: str = "google/gemma-3-4b-it",
        huggingface_token: str = None,
        max_history: int = 8,
        max_chunks: int = 5,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        """
        Инициализация бота
        
        Args:
            telegram_token: Токен бота в Telegram
            index_path: Путь к файлу индекса FAISS
            chunks_data_path: Путь к файлу с чанками и ссылками
            llm_model_name: Название модели для генерации ответов
            huggingface_token: Токен для доступа к моделям Hugging Face
            max_history: Максимальная длина истории разговора
            max_chunks: Максимальное количество чанков для ответа
            max_tokens: Максимальное количество токенов в ответе
            temperature: Температура генерации
            top_p: Параметр top_p для генерации
        """
        self.telegram_token = telegram_token
        self.index_path = index_path
        self.chunks_data_path = chunks_data_path
        self.llm_model_name = llm_model_name
        self.huggingface_token = huggingface_token
        self.max_history = max_history
        self.max_chunks = max_chunks
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # Хранилище истории разговоров для каждого пользователя
        self.user_conversations: Dict[int, ConversationMemory] = {}
        
        # Индикатор инициализации компонентов бота
        self.is_initialized = False
        
        # Статистика использования
        self.stats = {
            "total_queries": 0,
            "legal_queries": 0,
            "start_time": datetime.now()
        }
    
    async def initialize(self):
        """
        Асинхронная инициализация компонентов бота (retriever и generator)
        """
        if self.is_initialized:
            logger.info("Бот уже инициализирован")
            return
        
        try:
            # Авторизация в Hugging Face Hub
            if self.huggingface_token:
                login(token=self.huggingface_token)
                logger.info("Авторизация в Hugging Face Hub успешна")
            
            # Инициализация компонентов
            logger.info("Инициализация LegalRetriever...")
            self.retriever = LegalRetriever(
                index_path=self.index_path,
                chunks_data_path=self.chunks_data_path,
                top_k=self.max_chunks
            )
            
            logger.info("Инициализация LegalAnswerGenerator...")
            self.generator = LegalAnswerGenerator(
                model_name=self.llm_model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                huggingface_token=self.huggingface_token,
                max_chunks=self.max_chunks
            )
            
            self.is_initialized = True
            logger.info("Инициализация бота завершена успешно")
            
        except Exception as e:
            logger.error(f"Ошибка при инициализации бота: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_conversation_memory(self, user_id: int) -> ConversationMemory:
        """
        Получение или создание объекта ConversationMemory для пользователя
        
        Args:
            user_id: ID пользователя в Telegram
            
        Returns:
            Объект ConversationMemory для пользователя
        """
        if user_id not in self.user_conversations:
            self.user_conversations[user_id] = ConversationMemory(max_history=self.max_history)
        return self.user_conversations[user_id]
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Обработка команды /start
        """
        user_id = update.effective_user.id
        username = update.effective_user.username or "пользователь"
        
        # Создаем новую историю разговора для пользователя
        self.user_conversations[user_id] = ConversationMemory(max_history=self.max_history)
        
        # Отправляем приветственное сообщение
        welcome_message = (
            f"Здравствуйте, {username}! 👋\n\n"
            "Я - юридический ассистент LegalGuardian. Могу помочь вам найти ответы на вопросы, "
            "связанные с российским законодательством.\n\n"
            "Вы можете задавать вопросы о:\n"
            "• правах и обязанностях граждан\n"
            "• нормах и положениях законов\n"
            "• юридических процедурах\n"
            "• налогообложении\n"
            "• трудовом, семейном, жилищном, административном праве\n\n"
            "Для очистки истории разговора используйте команду /clear.\n"
            "Чтобы получить справку, введите /help.\n\n"
            "Пожалуйста, задайте ваш вопрос."
        )
        
        await update.message.reply_text(welcome_message)
        logger.info(f"Новый пользователь: {user_id} ({username})")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Обработка команды /help
        """
        help_message = (
            "🔍 *LegalGuardian - юридический ассистент*\n\n"
            "*Доступные команды:*\n"
            "/start - Начать работу с ботом\n"
            "/help - Показать эту справку\n"
            "/clear - Очистить историю разговора\n\n"
            
            "*Как использовать:*\n"
            "• Просто задавайте вопросы, связанные с российским законодательством\n"
            "• Я буду отвечать, опираясь на актуальные правовые нормы\n"
            "• Вы можете задавать уточняющие вопросы в рамках диалога\n\n"
            
            "*Ограничения:*\n"
            "• Я не могу предоставлять индивидуальные юридические консультации\n"
            "• Мои ответы не заменяют консультацию профессионального юриста\n"
            "• Я работаю с общими нормами законодательства\n\n"
            
            "При сложных юридических вопросах рекомендую обратиться к квалифицированному юристу."
        )
        
        await update.message.reply_text(help_message, parse_mode="Markdown")
    
    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Обработка команды /clear - очистка истории разговора
        """
        user_id = update.effective_user.id
        
        # Очищаем историю
        if user_id in self.user_conversations:
            self.user_conversations[user_id].clear()
        else:
            self.user_conversations[user_id] = ConversationMemory(max_history=self.max_history)
        
        await update.message.reply_text("История разговора очищена. Вы можете начать новую беседу.")
        logger.info(f"История разговора очищена для пользователя {user_id}")
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Обработка команды /stats - показ статистики бота
        """
        uptime = datetime.now() - self.stats["start_time"]
        uptime_str = str(uptime).split('.')[0]  # Удаляем миллисекунды
        
        legal_percentage = 0
        if self.stats["total_queries"] > 0:
            legal_percentage = (self.stats["legal_queries"] / self.stats["total_queries"]) * 100
        
        stats_message = (
            "📊 *Статистика бота*\n\n"
            f"Время работы: {uptime_str}\n"
            f"Всего запросов: {self.stats['total_queries']}\n"
            f"Юридических запросов: {self.stats['legal_queries']} ({legal_percentage:.1f}%)\n"
            f"Активных пользователей: {len(self.user_conversations)}\n"
        )
        
        await update.message.reply_text(stats_message, parse_mode="Markdown")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Обработка текстовых сообщений от пользователя
        """
        if not self.is_initialized:
            await update.message.reply_text(
                "Бот инициализируется, пожалуйста, подождите немного и повторите запрос..."
            )
            await self.initialize()
            return
        
        user_id = update.effective_user.id
        query = update.message.text
        
        # Обновляем статистику
        self.stats["total_queries"] += 1
        
        # Получаем или создаем историю разговора для пользователя
        conversation_memory = self.get_conversation_memory(user_id)
        
        # Отправляем индикатор набора текста
        await update.message.chat.send_action("typing")
        
        try:
            # Проверяем, является ли запрос юридическим вопросом
            is_legal = self.retriever.is_legal_question(query)
            
            if not is_legal:
                await update.message.reply_text(
                    "Извините, я могу отвечать только на юридические вопросы, связанные с российским законодательством. "
                    "Пожалуйста, задайте вопрос, касающийся правовых норм, законов или юридических процедур."
                )
                logger.info(f"Неюридический запрос от пользователя {user_id}: '{query}'")
                return
            
            # Обновляем статистику юридических запросов
            self.stats["legal_queries"] += 1
            
            # Поиск релевантных документов
            retrieved_chunks = self.retriever.search(query, is_legal_question=True)
            
            # Если ничего не найдено
            if not retrieved_chunks:
                await update.message.reply_text(
                    "К сожалению, я не нашел релевантной информации по вашему запросу в моей базе знаний. "
                    "Попробуйте переформулировать вопрос или задать более конкретный запрос."
                )
                logger.info(f"Нет результатов поиска для запроса пользователя {user_id}: '{query}'")
                return
            
            # Генерация ответа
            answer = self.generator.generate_answer(
                user_query=query,
                retrieved_chunks=retrieved_chunks,
                conversation_memory=conversation_memory
            )
            
            # Проверка качества ответа
            if not self.generator.is_legal_answer(query, answer):
                # Если ответ не прошел проверку качества
                await update.message.reply_text(
                    "Извините, я не смог сформировать качественный ответ на основе имеющейся у меня информации. "
                    "Попробуйте задать более конкретный вопрос или уточнить, что именно вас интересует."
                )
                logger.warning(f"Ответ низкого качества для пользователя {user_id}: '{query}'")
                return
            
            # Отправка ответа пользователю
            await update.message.reply_text(answer)
            logger.info(f"Ответ отправлен пользователю {user_id}")
            
        except Exception as e:
            logger.error(f"Ошибка при обработке сообщения от пользователя {user_id}: {e}")
            await update.message.reply_text(
                "Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже или задайте другой вопрос."
            )
    
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """
        Обработка ошибок в обработчиках сообщений
        """
        logger.error(f"Ошибка в обработчике: {context.error}")
        if update and isinstance(update, Update) and update.effective_chat:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Произошла ошибка при обработке запроса. Пожалуйста, попробуйте позже."
            )
    
    def run(self):
        """
        Запуск бота
        """
        # Создание и настройка приложения бота
        application = Application.builder().token(self.telegram_token).build()
        
        # Инициализация компонентов бота
        asyncio.run(self.initialize())
        
        # Регистрация обработчиков команд
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("clear", self.clear_command))
        application.add_handler(CommandHandler("stats", self.stats_command))
        
        # Регистрация обработчика текстовых сообщений
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Регистрация обработчика ошибок
        application.add_error_handler(self.error_handler)
        
        # Запуск бота
        logger.info("Запуск бота...")
        application.run_polling()


def initialize_bot(
    telegram_token: str = None,
    index_path: str = None,
    chunks_data_path: str = None,
    llm_model_name: str = "google/gemma-3-4b-it",
    huggingface_token: str = None
):
    """
    Инициализация и запуск Telegram бота
    
    Args:
        telegram_token: Токен бота в Telegram
        index_path: Путь к файлу индекса FAISS
        chunks_data_path: Путь к файлу с чанками и ссылками
        llm_model_name: Название модели для генерации ответов
        huggingface_token: Токен для доступа к моделям Hugging Face
    
    Returns:
        Экземпляр бота
    """
    # Загрузка переменных окружения
    load_dotenv()
    
    # Получение токенов и путей из переменных окружения, если не указаны явно
    telegram_token = telegram_token or os.environ.get("TELEGRAM_TOKEN")
    index_path = index_path or os.environ.get("INDEX_PATH")
    chunks_data_path = chunks_data_path or os.environ.get("CHUNKS_DATA_PATH")
    huggingface_token = huggingface_token or os.environ.get("HUGGINGFACE_TOKEN")
    
    # Проверка наличия обязательных параметров
    if not telegram_token:
        raise ValueError("Не указан токен Telegram. Укажите его через аргумент или в переменной окружения TELEGRAM_TOKEN.")
    
    if not index_path:
        raise ValueError("Не указан путь к индексу FAISS. Укажите его через аргумент или в переменной окружения INDEX_PATH.")
    
    if not chunks_data_path:
        raise ValueError("Не указан путь к файлу с чанками. Укажите его через аргумент или в переменной окружения CHUNKS_DATA_PATH.")
    
    # Создание бота
    try:
        bot = LegalGuardianBot(
            telegram_token=telegram_token,
            index_path=index_path,
            chunks_data_path=chunks_data_path,
            llm_model_name=llm_model_name,
            huggingface_token=huggingface_token,
            max_history=int(os.environ.get("MAX_HISTORY", 8)),
            max_chunks=int(os.environ.get("MAX_CHUNKS", 5)),
            max_tokens=int(os.environ.get("MAX_TOKENS", 1024)),
            temperature=float(os.environ.get("TEMPERATURE", 0.7)),
            top_p=float(os.environ.get("TOP_P", 0.9))
        )
        
        logger.info("Бот успешно инициализирован")
        return bot
        
    except Exception as e:
        logger.error(f"Ошибка при инициализации бота: {e}")
        import traceback
        traceback.print_exc()
        raise


def run_bot():
    """
    Инициализация и запуск бота
    """
    try:
        # Инициализация бота
        bot = initialize_bot()
        
        # Запуск бота
        bot.run()
        
    except Exception as e:
        logger.error(f"Ошибка при запуске бота: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_bot() 