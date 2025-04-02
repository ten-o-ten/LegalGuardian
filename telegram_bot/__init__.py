#!/usr/bin/env python
"""
Пакет telegram_bot для юридического ассистента LegalGuardian.

Этот пакет содержит модули для:
- Поиска релевантной информации в правовых документах (retriever.py)
- Генерации ответов на основе найденных документов (generator.py)
- Взаимодействия с пользователями через Telegram (bot.py)
"""

from .retriever import LegalRetriever
from .generator import LegalAnswerGenerator, ConversationMemory
from .bot import LegalGuardianBot, initialize_bot, run_bot

__all__ = [
    'LegalRetriever',
    'LegalAnswerGenerator',
    'ConversationMemory',
    'LegalGuardianBot',
    'run_bot',
]

__version__ = "0.1.0"
__author__ = "LegalGuardian Team" 