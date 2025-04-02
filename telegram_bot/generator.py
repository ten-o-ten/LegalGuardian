#!/usr/bin/env python3
"""
Модуль для генерации ответов на юридические вопросы
с использованием языковой модели и контекста из
релевантных юридических документов.
"""

import logging
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List, Dict, Any, Optional
from huggingface_hub import login
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class ConversationMemory:
    """
    Класс для управления историей разговора
    """
    
    def __init__(self, max_history: int = 8):
        """
        Инициализация памяти разговора
        
        Args:
            max_history: максимальное количество последних сообщений для хранения
        """
        self.messages = []
        self.max_history = max_history
    
    def add_message(self, role: str, content: str):
        """
        Добавление сообщения в историю
        
        Args:
            role: роль отправителя (user или assistant)
            content: содержание сообщения
        """
        # Проверка валидности роли
        if role not in ["user", "assistant"]:
            raise ValueError(f"Недопустимая роль: {role}. Используйте 'user' или 'assistant'")
        
        # Добавление сообщения
        self.messages.append({"role": role, "content": content})
        
        # Если история превысила максимальную длину, удаляем старые сообщения
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        Получение истории разговора
        
        Returns:
            Список сообщений в формате [{"role": role, "content": content}, ...]
        """
        return self.messages.copy()
    
    def clear(self):
        """
        Очистка истории разговора
        """
        self.messages = []


class LegalAnswerGenerator:
    """
    Генератор ответов на юридические вопросы на основе LLM
    """
    
    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        huggingface_token: Optional[str] = None,
        max_chunks: int = 5
    ):
        """
        Инициализация генератора ответов
        
        Args:
            model_name: название модели для генерации
            max_tokens: максимальное количество токенов в ответе
            temperature: температура сэмплирования (высокие значения дают более разнообразные ответы)
            top_p: вероятность отсечения (nucleus sampling)
            huggingface_token: токен для доступа к моделям Hugging Face
            max_chunks: максимальное количество чанков для использования в контексте
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.huggingface_token = huggingface_token or os.environ.get("HUGGINGFACE_TOKEN")
        self.max_chunks = max_chunks
        
        # Инициализация модели
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Инициализация языковой модели
        """
        # Авторизация в Hugging Face Hub, если предоставлен токен
        if self.huggingface_token:
            try:
                login(token=self.huggingface_token)
                logger.info("Успешная авторизация в Hugging Face Hub")
            except Exception as e:
                logger.error(f"Ошибка авторизации в Hugging Face Hub: {e}")
                raise
        
        # Инициализация токенизатора и модели
        try:
            logger.info(f"Загрузка модели: {self.model_name}")
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Используется устройство: {self.device}")
            
            # Инициализация токенизатора
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Инициализация модели
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            logger.info(f"Модель {self.model_name} успешно загружена")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise
    
    def _prepare_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Подготовка контекста для генерации ответа
        
        Args:
            chunks: релевантные чанки с их метаданными
            
        Returns:
            Контекст для модели
        """
        # Ограничиваем количество используемых чанков
        used_chunks = chunks[:min(len(chunks), self.max_chunks)]
        
        # Формируем контекст, включая ссылки на источники
        context = "Информация из правовых источников:\n\n"
        
        for i, chunk in enumerate(used_chunks):
            context += f"[Документ {i+1}] {chunk['reference']}\n"
            context += f"{chunk['chunk']}\n\n"
        
        return context
    
    def _prepare_system_prompt(self) -> str:
        """
        Подготовка системного промпта для LLM
        
        Returns:
            Системный промпт
        """
        return """Ты - юридический ассистент, отвечающий на вопросы, связанные с российским законодательством.
Твоя задача - предоставлять точную и полезную информацию, основанную на правовых источниках.

Следуй этим правилам при составлении ответов:
1. Основывай свои ответы на предоставленной информации из правовых источников
2. Цитируй конкретные законы, статьи и нормативные акты, когда это возможно
3. Отвечай только на юридические вопросы
4. Указывай источники информации в конце ответа
5. Если информации недостаточно, признай это и предложи, где пользователь может найти дополнительную информацию
6. Не давай юридических советов, которые могут рассматриваться как профессиональная юридическая консультация
7. Используй ясный и понятный язык, избегая излишне сложной юридической терминологии
8. Если задан вопрос не по юридической тематике, вежливо объясни, что ты специализируешься только на юридических вопросах

Ты должен отвечать на русском языке, даже если вопрос задан на другом языке."""
    
    def _format_chat_messages(self, system_prompt: str, conversation_history: List[Dict[str, str]], user_query: str, context: str) -> List[Dict[str, str]]:
        """
        Форматирование сообщений для чата
        
        Args:
            system_prompt: системный промпт
            conversation_history: история беседы
            user_query: запрос пользователя
            context: контекст из релевантных документов
            
        Returns:
            Список сообщений для модели
        """
        # Начинаем с системного сообщения
        messages = [{"role": "user", "content": system_prompt}, {"role": "assistant", "content": "Я готов помочь с юридическими вопросами по российскому законодательству."}]
        
        # Добавляем историю разговора
        for message in conversation_history:
            # Проверка чередования ролей
            if messages[-1]["role"] == message["role"]:
                logger.warning(f"Нарушение чередования ролей. Пропуск сообщения: {message}")
                continue
            messages.append(message)
        
        # Инструкция по использованию контекста и запрос пользователя
        context_message = f"""Используй следующую информацию из российских правовых источников для ответа на вопрос.
        
{context}

Вопрос: {user_query}"""
        
        # Добавляем финальное сообщение пользователя
        if messages[-1]["role"] == "user":
            # Если последнее сообщение - от пользователя, обновляем его контекстом
            messages[-1]["content"] += f"\n\n{context_message}"
        else:
            # Иначе добавляем новое сообщение пользователя
            messages.append({"role": "user", "content": context_message})
        
        return messages
    
    def generate_answer(
        self,
        user_query: str,
        retrieved_chunks: List[Dict[str, Any]],
        conversation_memory: ConversationMemory
    ) -> str:
        """
        Генерация ответа на юридический вопрос
        
        Args:
            user_query: запрос пользователя
            retrieved_chunks: релевантные чанки из индекса
            conversation_memory: объект для хранения истории разговора
            
        Returns:
            Ответ на вопрос
        """
        try:
            # Подготовка данных для модели
            system_prompt = self._prepare_system_prompt()
            context = self._prepare_context(retrieved_chunks)
            conversation_history = conversation_memory.get_history()
            
            # Форматирование сообщений для чата
            messages = self._format_chat_messages(
                system_prompt=system_prompt,
                conversation_history=conversation_history,
                user_query=user_query,
                context=context
            )
            
            # Генерация ответа
            logger.info(f"Генерация ответа на запрос: '{user_query}'")
            
            # Преобразование сообщений в формат, понятный модели
            model_inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Запуск генерации
            response_ids = self.model.generate(
                model_inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Декодирование результата
            response = self.tokenizer.decode(
                response_ids[0][model_inputs.shape[1]:],
                skip_special_tokens=True
            )
            
            # Если ответ содержит только пробельные символы или слишком короткий
            if not response.strip() or len(response.strip()) < 10:
                logger.warning("Получен пустой или слишком короткий ответ, генерация запасного ответа")
                response = "К сожалению, не удалось сформировать ответ на основе имеющейся информации. Рекомендую обратиться к профессиональному юристу для получения квалифицированной консультации по этому вопросу."
            
            # Добавляем вопрос пользователя и ответ в историю
            conversation_memory.add_message("user", user_query)
            conversation_memory.add_message("assistant", response)
            
            logger.info("Ответ сгенерирован успешно")
            return response
            
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            import traceback
            traceback.print_exc()
            
            # В случае ошибки возвращаем стандартный ответ
            error_response = "Извините, произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте переформулировать вопрос или задать его позже."
            conversation_memory.add_message("user", user_query)
            conversation_memory.add_message("assistant", error_response)
            
            return error_response
    
    def is_legal_answer(self, query: str, response: str) -> bool:
        """
        Проверка, является ли ответ подходящим для юридического вопроса
        
        Args:
            query: запрос пользователя
            response: ответ модели
            
        Returns:
            True, если ответ адекватный для юридического вопроса
        """
        # Проверяем минимальную длину ответа
        if len(response.split()) < 15:
            logger.warning(f"Ответ слишком короткий: '{response}'")
            return False
        
        # Проверяем наличие отказа отвечать
        refusal_phrases = [
            "не могу дать юридическую консультацию",
            "не могу предоставить юридическую консультацию",
            "я не юрист",
            "не могу дать профессиональный совет",
            "обратитесь к юристу",
            "я не могу комментировать",
            "не относится к юридической тематике",
            "не в моей компетенции"
        ]
        
        for phrase in refusal_phrases:
            if phrase in response.lower() and len(response.split()) < 50:
                logger.warning(f"Ответ содержит отказ: '{response}'")
                return False
        
        # Проверка наличия юридической терминологии в ответе
        legal_terms = [
            "закон", "право", "статья", "кодекс", "законодательств",
            "норматив", "постановлен", "суд", "юридическ", "правоотношен",
            "федеральн", "нормативно-правов"
        ]
        
        has_legal_terms = any(term in response.lower() for term in legal_terms)
        if not has_legal_terms:
            logger.warning(f"Ответ не содержит юридической терминологии: '{response}'")
            return False
        
        return True

    def format_answer_with_sources(self, answer, context_documents):
        """
        Форматирует ответ с источниками информации
        
        Args:
            answer (str): Сгенерированный ответ
            context_documents (list): Список релевантных документов
        
        Returns:
            str: Отформатированный ответ с источниками
        """
        # Если нет источников или их недостаточно, возвращаем только ответ
        if not context_documents or len(context_documents) == 0:
            return answer
        
        # Добавляем источники
        sources_text = "\n\n📚 Источники информации:\n"
        added_refs = set()  # Для отслеживания уникальных источников
        
        for i, doc in enumerate(context_documents):
            if "reference" in doc and doc["reference"]:
                ref = doc["reference"].strip()
                # Добавляем только уникальные источники
                if ref not in added_refs:
                    sources_text += f"{i+1}. {ref}\n"
                    added_refs.add(ref)
        
        # Возвращаем ответ с источниками, если есть уникальные источники
        if len(added_refs) > 0:
            return f"{answer}\n{sources_text}"
        else:
            return answer 