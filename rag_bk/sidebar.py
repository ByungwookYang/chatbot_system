import streamlit as st
from langchain_core.prompts import load_prompt, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from rag_bk.modules.tools import WebSearchTool, retriever_tool
from rag_bk.modules.agent import create_agent_executor
from rag_bk.bk_messages import random_uuid


def show_sidebar():
    """
    Streamlit 사이드바에 개인정보 입력 + 제출 버튼을 구성하고,
    버튼 클릭 시 st.session_state에 특정 정보를 저장합니다.
    """
    # 사이드바 내부에 배치
    with st.sidebar:
        # 초기화 버튼
        clear_btn = st.button("대화 초기화")

        st.subheader("간단한 개인정보를 입력해주세요.")

        # 나이 입력
        age = st.number_input(
            "나이",
            min_value=0,
            max_value=120,
            value=0,
        )

        # 성별 입력
        gender = st.selectbox(
            "성별",
            ("남자", "여자"),
            index=None,
            placeholder="성별을 선택하세요...",
        )

        # 결혼 여부
        married = st.radio(
            "기혼여부",
            ["기혼💍", "미혼👨🏻‍💻", "재혼💛", "기타"],
            index=None,
        )

        # 자녀 정보
        options = st.multiselect(
            "자녀정보",
            [
                "없음",
                "아들",
                "딸",
                "1명",
                "2명",
                "3명",
                "0-5세",
                "6-10세",
                "11-15세",
                "16-20세",
                "20세 이상",
            ],
            default=None,
        )

        # 가족과의 유대감 점수
        family_score = st.slider(
            "가족과의 유대감 점수",
            min_value=0,
            max_value=10,
            step=1,
        )

        # 추가 상담 요청 내용
        user_text_prompt = st.text_area(
            "상담 요청 내용",
            "폐암에 대한 상담 요청",
            height=100,
        )

        # 제출 버튼
        apply_btn = st.button("제출", key="primary")

        # 제출 버튼 로직
        if apply_btn:
            with st.spinner("챗봇상담사를 지정중입니다..."):
                # 상담자 답변 읽어오기
                age_val = st.session_state.get("age", age)
                gender_val = st.session_state.get("gender", gender)
                married_val = st.session_state.get("married", married)
                options_val = st.session_state.get("options", options)
                family_score_val = st.session_state.get("family_score", family_score)

                # step 1 페르소나 부여 프롬프트 생성(LLM을 통한)
                # gen_prom.yaml 로드(개인정보에 맞는 페르소나를 부여하기 위한 프롬프트)
                loaded_prompt = load_prompt("prompts/gen_prom.yaml", encoding="utf-8")

                # 최종 페르소나 부여 프롬프트 생성
                final_template = (
                    f"{user_text_prompt}, {loaded_prompt.template}, "  # 사용자 상담내용 + 개인정보에 맞는 페르소나를 부여하기 위한 프롬프트
                    f"사용자의 나이는 {age_val}, 성별은 {gender_val}, "  # 사용자 개인정보
                    f"결혼여부는 {married_val}, 자녀정보는 {options_val}, "
                    f"가족과의 유대감은 {family_score_val}"
                )

                # LLM 호출 (첫 번째 체인)
                prompt1 = PromptTemplate.from_template(template=final_template)
                llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2)
                chain1 = prompt1 | llm | StrOutputParser()
                st.session_state["new_prompt"] = chain1.invoke("")  # 페르소나 생성

                # step 2 부여된 페르소나를 이용한 최종 답변 LLM 생성
                tool1 = retriever_tool()

                # WebSearchTool 인스턴스 생성
                tool2 = WebSearchTool().create()

                # 리액트형 에이전트 생성
                st.session_state["react_agent"] = create_agent_executor(
                    model_name="gpt-4o-mini",
                    tools=[tool1, tool2],
                )
                # 고유 스레드 ID
                st.session_state["thread_id"] = random_uuid()

                st.success("제출이 완료되었습니다! 상담을 진행하세요.")
        # 초기화 버튼이 눌리면...
    if clear_btn:
        st.session_state["messages"] = []
        st.session_state["thread_id"] = random_uuid()
