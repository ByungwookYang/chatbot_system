from attr import dataclass
import streamlit as st
from langchain_core.messages.chat import ChatMessage
from rag_bk.modules.handler import format_search_result


@dataclass
class ChatMessageWithType:
    chat_message: ChatMessage
    msg_type: str
    tool_name: str


# 이전 대화를 출력
def print_messages():
    for message in st.session_state["messages"]:
        if message.msg_type == "text":
            st.chat_message(message.chat_message.role).write(
                message.chat_message.content
            )
        elif message.msg_type == "tool_result":
            with st.expander(f"✅ {message.tool_name}"):
                st.markdown(message.chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message, msg_type="text", tool_name=""):
    if msg_type == "text":
        st.session_state["messages"].append(
            ChatMessageWithType(
                chat_message=ChatMessage(role=role, content=message),
                msg_type="text",
                tool_name=tool_name,
            )
        )
    elif msg_type == "tool_result":
        st.session_state["messages"].append(
            ChatMessageWithType(
                chat_message=ChatMessage(
                    role="assistant", content=format_search_result(message)
                ),
                msg_type="tool_result",
                tool_name=tool_name,
            )
        )
