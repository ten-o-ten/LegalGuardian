#!/usr/bin/env python3
"""
Модуль для поиска релевантной информации в индексе FAISS.
Предоставляет класс LegalRetriever для поиска релевантных документов
по запросу пользователя.
"""

import os
import pickle
import faiss
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoTokenizer, AutoModel

# Настройка логирования
import logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class LegalRetriever:
    """
    Класс для поиска релевантных юридических документов по запросу
    """
    
    def __init__(
        self,
        index_path: str,
        chunks_data_path: str,
        embedding_model: Optional[str] = None,
        use_query_expansion: bool = True,
        top_k: int = 5
    ):
        """
        Инициализация ретривера
        
        Args:
            index_path: путь к файлу индекса FAISS
            chunks_data_path: путь к файлу с чанками и ссылками
            embedding_model: название модели для эмбеддингов (если None, будет взято из файла с чанками)
            use_query_expansion: использовать ли расширение запроса
            top_k: количество топ результатов для возврата
        """
        self.index_path = index_path
        self.chunks_data_path = chunks_data_path
        self.use_query_expansion = use_query_expansion
        self.top_k = top_k
        
        # Загрузка индекса и данных
        self._load_index_and_data()
        
        # Инициализация модели для эмбеддингов
        self.embedding_model = embedding_model or self.chunks_data.get("embedder_model", "intfloat/multilingual-e5-small")
        self._initialize_embedding_model()
    
    def _load_index_and_data(self):
        """
        Загрузка индекса FAISS и данных чанков
        """
        logger.info(f"Загрузка индекса из {self.index_path}")
        
        # Проверка наличия файлов
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Файл индекса не найден: {self.index_path}")
        
        if not os.path.exists(self.chunks_data_path):
            raise FileNotFoundError(f"Файл с данными чанков не найден: {self.chunks_data_path}")
        
        # Загрузка индекса
        try:
            self.index = faiss.read_index(self.index_path)
            logger.info(f"Индекс успешно загружен, содержит {self.index.ntotal} векторов")
        except Exception as e:
            logger.error(f"Ошибка при загрузке индекса: {e}")
            raise
        
        # Загрузка данных чанков
        try:
            with open(self.chunks_data_path, "rb") as f:
                self.chunks_data = pickle.load(f)
            
            self.chunks = self.chunks_data["chunks"]
            self.references = self.chunks_data["references"]
            
            logger.info(f"Данные чанков успешно загружены, {len(self.chunks)} чанков")
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных чанков: {e}")
            raise
    
    def _initialize_embedding_model(self):
        """
        Инициализация модели для создания эмбеддингов
        """
        logger.info(f"Инициализация модели эмбеддингов: {self.embedding_model}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
            self.model = AutoModel.from_pretrained(self.embedding_model).to(self.device)
            self.model.eval()
            logger.info(f"Модель эмбеддингов успешно загружена на {self.device}")
        except Exception as e:
            logger.error(f"Ошибка при инициализации модели эмбеддингов: {e}")
            raise
    
    def _average_pool(self, last_hidden_states, attention_mask):
        """
        Усреднение токенов с учетом маски внимания
        """
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def _embed_query(self, query: str) -> np.ndarray:
        """
        Создание эмбеддинга запроса
        
        Args:
            query: текст запроса
            
        Returns:
            Эмбеддинг запроса
        """
        # Подготовка запроса в формате, подходящем для модели (для E5)
        processed_query = f"query: {query}"
        
        # Токенизация
        inputs = self.tokenizer(
            processed_query,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Создание эмбеддинга
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._average_pool(outputs.last_hidden_state, inputs["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
        # Преобразование в numpy массив
        query_embedding = embeddings.cpu().numpy()
        
        return query_embedding
    
    def _expand_query(self, query: str) -> str:
        """
        Расширение запроса для улучшения поиска
        
        Args:
            query: исходный запрос пользователя
            
        Returns:
            Расширенный запрос
        """
        # Добавление юридических терминов и фраз
        expansions = [
            "юридические аспекты",
            "правовые нормы",
            "законодательство",
            "федеральный закон",
            "права и обязанности",
            "правовой статус",
            "юридическое понятие",
            "согласно закону",
            "нормативно-правовой акт"
        ]
        
        # Поиск и удаление из запроса фраз, не относящихся к юридическим вопросам
        remove_phrases = [
            "скажи мне",
            "расскажи о",
            "что такое",
            "как понять",
            "объясни",
            "можешь ли ты",
            "пожалуйста",
            "подскажи"
        ]
        
        expanded_query = query
        
        # Удаление нерелевантных фраз
        for phrase in remove_phrases:
            expanded_query = expanded_query.replace(phrase, "")
        
        # Поиск наиболее релевантного расширения
        best_expansion = None
        max_overlap = -1
        
        for expansion in expansions:
            # Простая метрика - количество общих слов
            query_words = set(query.lower().split())
            expansion_words = set(expansion.lower().split())
            overlap = len(query_words.intersection(expansion_words))
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_expansion = expansion
        
        # Если запрос слишком короткий, добавляем юридический контекст
        if len(expanded_query.split()) < 3:
            expanded_query += f" {best_expansion}"
        
        # Очистка лишних пробелов
        expanded_query = " ".join(expanded_query.split())
        
        if expanded_query != query:
            logger.info(f"Запрос расширен: '{query}' -> '{expanded_query}'")
        
        return expanded_query
    
    def search(self, query: str, is_legal_question: bool = True) -> List[Dict[str, Any]]:
        """
        Поиск релевантных документов по запросу
        
        Args:
            query: запрос пользователя
            is_legal_question: является ли запрос юридическим вопросом
            
        Returns:
            Список словарей с релевантными документами и их метаданными
        """
        if not is_legal_question:
            logger.info(f"Запрос не является юридическим вопросом: '{query}'")
            return []
        
        try:
            # Расширение запроса, если включено
            if self.use_query_expansion:
                expanded_query = self._expand_query(query)
            else:
                expanded_query = query
            
            # Создание эмбеддинга запроса
            query_embedding = self._embed_query(expanded_query)
            
            # Нормализация вектора
            faiss.normalize_L2(query_embedding)
            
            # Поиск ближайших соседей
            scores, indices = self.index.search(query_embedding, self.top_k)
            
            # Формирование результатов
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                score = float(scores[0][i])
                
                # Проверка валидности индекса
                if idx < 0 or idx >= len(self.chunks):
                    continue
                
                # Формирование результата
                result = {
                    "chunk": self.chunks[idx],
                    "reference": self.references[idx],
                    "score": score
                }
                results.append(result)
            
            logger.info(f"Найдено {len(results)} релевантных документов для запроса: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка при поиске документов: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def is_legal_question(self, query: str) -> bool:
        """
        Определение, является ли запрос юридическим вопросом
        
        Args:
            query: запрос пользователя
            
        Returns:
            True, если запрос является юридическим вопросом
        """
        # Ключевые слова юридической тематики
        legal_keywords = [
            "закон", "право", "юрид", "кодекс", "статья", "суд", "догов", "норм",
            "ответств", "регул", "легал", "законодат", "обязан", "регистрац",
            "защит", "патент", "лиценз", "штраф", "санкц", "иск", "налог",
            "имуществ", "наслед", "собствен", "возмещ", "компенс", "претенз",
            "нотари", "адвокат", "доверен", "учред", "устав", "акционер", "директор"
        ]
        
        # Проверка наличия ключевых слов в запросе
        query_lower = query.lower()
        
        for keyword in legal_keywords:
            if keyword in query_lower:
                logger.info(f"Запрос определен как юридический (ключевое слово: '{keyword}'): '{query}'")
                return True
        
        # Дополнительная проверка на наличие вопросов о правах, обязанностях и т.д.
        legal_patterns = [
            "имею ли я право", "можно ли", "законно ли", "правомерно ли",
            "как правильно", "какие права", "какие обязанности", "что делать если",
            "как оформить", "как получить", "как подать", "как заполнить",
            "как составить", "как зарегистрировать", "что говорит закон",
            "что сказано в законе", "по закону", "согласно закону"
        ]
        
        for pattern in legal_patterns:
            if pattern in query_lower:
                logger.info(f"Запрос определен как юридический (паттерн: '{pattern}'): '{query}'")
                return True
        
        logger.info(f"Запрос не определен как юридический: '{query}'")
        return False 