from collections import defaultdict, deque

class ConversationMemory:
    """Класс для хранения истории сообщений пользователей"""
    
    def __init__(self, max_messages=10):
        """
        Инициализация памяти разговоров
        
        Args:
            max_messages: максимальное количество сообщений для хранения на пользователя
        """
        self.max_messages = max_messages
        # Используем defaultdict(deque) для эффективного хранения истории сообщений
        self.memory = defaultdict(lambda: deque(maxlen=max_messages))
    
    def add_message(self, user_id, role, text):
        """
        Добавление сообщения в историю
        
        Args:
            user_id: идентификатор пользователя
            role: роль ('user' или 'assistant')
            text: текст сообщения
        """
        self.memory[user_id].append({
            "role": role,
            "content": [{"type": "text", "text": text}]
        })
    
    def get_conversation_history(self, user_id):
        """
        Получение истории разговора для пользователя
        
        Args:
            user_id: идентификатор пользователя
            
        Returns:
            Список сообщений в формате, готовом для использования с моделью
        """
        return list(self.memory[user_id])
    
    def get_last_n_messages(self, user_id, n=None):
        """
        Получение последних N сообщений для пользователя
        
        Args:
            user_id: идентификатор пользователя
            n: количество сообщений (если None, возвращает все сохраненные)
            
        Returns:
            Список последних N сообщений
        """
        if n is None:
            return list(self.memory[user_id])
        
        history = self.memory[user_id]
        if len(history) <= n:
            return list(history)
        
        return list(history)[-n:]
    
    def clear_history(self, user_id):
        """
        Очистка истории для пользователя
        
        Args:
            user_id: идентификатор пользователя
        """
        if user_id in self.memory:
            self.memory[user_id].clear()
    
    def get_formatted_history(self, user_id, n=None):
        """
        Получение форматированной истории для отладки
        
        Args:
            user_id: идентификатор пользователя
            n: количество сообщений (если None, возвращает все сохраненные)
            
        Returns:
            Строка с форматированной историей
        """
        messages = self.get_last_n_messages(user_id, n)
        if not messages:
            return "История сообщений пуста"
        
        formatted = []
        for msg in messages:
            role = msg["role"]
            text = msg["content"][0]["text"] if isinstance(msg["content"], list) else msg["content"]
            formatted.append(f"[{role.upper()}]: {text}")
        
        return "\n".join(formatted) 