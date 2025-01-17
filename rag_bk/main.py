import streamlit as st
from rag_bk.bk_logging import langsmith
from dotenv import load_dotenv
from rag_bk.modules.handler import stream_handler
from sidebar import show_sidebar
from st_function import print_messages, add_message

# API KEY 정보로드
load_dotenv()

# 프로젝트 이름
langsmith("챗봇상담")
import os

st.title("챗봇상담 💬")

# 대화기록을 저장하기 위한 용도로 생성
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ReAct Agent 초기화
if "react_agent" not in st.session_state:
    st.session_state["react_agent"] = None

# include_domains 초기화
if "include_domains" not in st.session_state:
    st.session_state["include_domains"] = []

# 사이드바 생성(sidebar.py로 생성)
show_sidebar()

# 이전 대화 기록 출력(st_function.py로 생성)
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 만약에 사용자 입력이 들어오면...
if user_input:
    agent = st.session_state["react_agent"]
    # Config 설정

    if agent is not None:
        config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
        # 사용자의 입력
        st.chat_message("user").write(user_input)

        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            container_messages, tool_args, agent_answer = stream_handler(
                container,
                agent,
                {
                    "messages": [
                        ("human", user_input),
                    ]
                },
                config,
            )

            # 대화기록을 저장한다.
            add_message("user", user_input)
            for tool_arg in tool_args:
                add_message(
                    "assistant",
                    tool_arg["tool_result"],
                    "tool_result",
                    tool_arg["tool_name"],
                )
            add_message("assistant", agent_answer)
    else:
        warning_msg.warning("사이드바에서 개인정보 입력을 완료해주세요.")
