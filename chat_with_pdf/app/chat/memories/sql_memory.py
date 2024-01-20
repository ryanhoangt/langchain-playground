from pydantic import BaseModel
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseChatMessageHistory
from app.chat.models import ChatArgs

from app.web.api import get_messages_by_conversation_id, add_message_to_conversation

class SqlMessageHistory(BaseChatMessageHistory, BaseModel):
    conversation_id: str

    @property
    def messages(self):
        return get_messages_by_conversation_id(self.conversation_id)

    def add_message(self, message):
        return add_message_to_conversation(
            self.conversation_id, 
            message.type, 
            message.content
        )
    
    def clear(self):
        pass

def build_memory(chat_args: ChatArgs):
    return ConversationBufferMemory(
        chat_memory=SqlMessageHistory(conversation_id=chat_args.conversation_id),
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )