import os
import json
import textwrap
import re
import argparse
from datetime import datetime
from typing import List, Any, Dict
import llm_client

def summarize_script(llm, script_path: str, script_content: str, summary_title: str):
    # 텍스트 길이 확인 및 청킹
    chunks = split_text(script_content)
    print(f"텍스트 길이: {len(script_content)}자 -> {len(chunks)}개 블록으로 분할")

    # 청크 여부 확인
    is_chunk = len(chunks) > 1
    
    # 부분 요약
    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"[{i+1}/{len(chunks)}] 요약 생성 중...")
        summary = summarize_chunk(llm, chunk, mode='part' if is_chunk else 'single')
        summaries.append(summary)
        print(f"--- 부분 요약 {i+1} ---\n{summary}\n")

    # 만약 청크가 여러 개였다면, 부분 요약본들을 모아서 최종 요약 수행 (Map-Reduce)
    if is_chunk:
        summary_text = "\n\n".join(summaries)
        final_summary = summarize_chunk(llm, summary_text, mode='final')
    else:
        summary_text = summaries[0]
        final_summary = summary_text

    print("\n" + "="*30)
    print(" [최종 요약 결과] ")
    print("="*30)
    print(final_summary)
    
    # 결과 파일 저장
    save_file(f"./scripts/final_{summary_title}", final_summary)

    # 요약본 저장
    save_file(f"./scripts/{summary_title}", summary_text)

    return summary_text

def split_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    오버랩 포함 텍스트 분할 함수
    - chunk_size: 한 덩어리 최대 길이
    - chunk_overlap: 겹치는 길이 (문맥 보존용)
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        # 끝점 계산: 현재 위치 + 청크 크기
        end = start + chunk_size
        
        # 텍스트 끝에 도달했으면 그냥 끝까지 다 넣음
        if end >= text_len:
            chunks.append(text[start:])
            break

        # [핵심 로직] 문장이 뚝 끊기지 않게 '줄바꿈'이나 '공백'을 찾아서 자름
        # end 위치 근처에서 가장 마지막 줄바꿈(\n) 위치를 찾음
        last_newline = text.rfind('\n', start, end)
        
        if last_newline != -1:
            # 줄바꿈이 있으면 거기서 자름 (깔끔)
            cut_point = last_newline + 1
        else:
            # 줄바꿈이 없으면 공백이라도 찾음
            last_space = text.rfind(' ', start, end)
            if last_space != -1:
                cut_point = last_space + 1
            else:
                # 공백도 없으면 그냥 강제로 자름
                cut_point = end

        # 덩어리 추가
        chunks.append(text[start:cut_point])

        # [Overlap 핵심] 
        # 다음 시작점은 '자른 위치'에서 '오버랩'만큼 뒤로 당겨서 잡음
        # 예: 1000자에서 잘랐고 오버랩이 100자면, 다음은 900자부터 시작
        start = cut_point - chunk_overlap
        
        # 만약 오버랩 때문에 무한루프 돌 것 같으면 강제로 전진 (안전장치)
        if start < 0: start = 0
    
    return chunks

def summarize_chunk(llm, text, mode='part'):
    """
    LLM을 사용하여 텍스트를 요약합니다.
    """

    # 프롬프트 설정
    if mode == 'part':  # chunk 여러개 부분 요약
        system_msg = "당신은 강의 내용을 논리적으로 분석하여 정리하는 '수석 연구원'입니다."
        raw_prompt = f"""
        [지시사항]
        제공된 [스크립트]를 읽고, 핵심 내용을 **상세한 노트 필기 형식**으로 정리하세요.
        단순한 사실 나열이 아니라, **'무엇이(What)', '왜(Why)', '어떻게(How)', '결과(Result)'**의 논리적 흐름이 보이도록 작성해야 합니다.

        1. **형식**: 반드시 `-(하이픈) **핵심키워드**: 상세 설명` 형식을 지키십시오.
        2. **내용 깊이**: 각 항목은 2~3문장(공백 포함 50~100자)으로 구체적으로 서술하십시오.
        3. **주의사항**: '강사는 말했다' 같은 표현을 배제하고, 객관적인 지식 정보만 추출하십시오.

        [작성 예시]
        - **메타버스 급성장**: 코로나19 팬데믹으로 인한 비대면 활동 증가가 기폭제가 되어, 5월 이후 관련 검색량이 폭발적으로 증가함.
        - **AI의 역설**: AI는 인간의 외로움을 달래주는 도구가 될 수 있으나, 과도한 의존 시 현실 인간관계의 단절이라는 부작용을 초래할 수 있음.
        - **카운터컬쳐**: 숏폼 콘텐츠가 주류가 되자 이에 대한 반작용으로 텍스트나 롱폼 영상에 집중하는 수요가 30% 비율로 생겨남.
        [스크립트]
        {text}

        ### [요약 결과]
        -"""
        
        # 공백 제거 처리
        user_msg = textwrap.dedent(raw_prompt).strip()

    elif mode == 'final':   # chunk 여러개 최종 요약
        system_msg = "당신은 복잡한 정보를 통찰력 있게 정리하여 보고서를 작성하는 '전문 에디터'입니다."
        user_msg = f"""
        다음은 영상의 핵심 내용을 정리한 노트들입니다. 
        이를 바탕으로 학습자를 위한 **구조화된 요약본**을 작성해주세요.

        [지시사항]
        1. **전체 요약:** 영상의 주제와 결론을 자연스러운 문장으로 서술하세요. (학습자가 '무엇을 배울 수 있는지' 알 수 있게)
        2. **분량:** 전체 길이는 공백 포함 **300~500자** 정도로 맞추세요.
        3. **스타일:** '영상에서는' 같은 표현을 빼고, 바로 지식을 전달하는 문체를 사용하세요.

        Response:
            (여기에 요약 문장 작성)

        [부분 요약본 내용]
        {text}
        """
        # user_msg = f"""
        # 다음은 영상의 핵심 내용을 정리한 노트들입니다. 
        # 이를 바탕으로 학습자를 위한 **구조화된 요약본**을 작성해주세요.

        # [지시사항]
        # 1. **전체 요약 (3문장 이내):** 영상의 주제와 결론을 자연스러운 문장으로 서술하세요. (학습자가 '무엇을 배울 수 있는지' 알 수 있게)
        # 2. **핵심 포인트 (5개 이내):** 영상에서 다루는 가장 중요한 개념이나 배움 포인트를 개조식(Bullet points)으로 정리하세요.
        # 3. **분량:** 전체 길이는 공백 포함 **300~500자** 정도로 맞추세요.
        # 4. **스타일:** '영상에서는' 같은 표현을 빼고, 바로 지식을 전달하는 문체를 사용하세요.

        # Response:
        #     ## 전체 요약
        #     (여기에 요약 문장 작성)

        #     ## 핵심 포인트
        #     - (포인트 1)
        #     - (포인트 2)
        #     ...

        # [부분 요약본 내용]
        # {text}
        # """

    else:   # mode = 'single' # chunk 1개 단일 요약
        system_msg = "당신은 복잡한 정보를 통찰력 있게 정리하여 교육 자료를 만드는 '전문 에디터'입니다."
        user_msg = f"""
        다음은 교육 영상의 전체 스크립트입니다.
        이를 종합하여 학습자를 위한 **완결성 있는 최종 요약보고서**를 작성해 주세요.

        [지시사항]
        1. **## 전체 요약**: 영상의 주제, 배경, 결론을 아우르는 3~4문장의 줄글로 작성하세요. (학습자가 배울 수 있는 가치 중심)
        2. **## 핵심 포인트**: 시험 문제로 출제될 만한 중요한 개념, 수치, 이론, 인과관계를 5~7개의 개조식(Bullet points)으로 정리하세요.
        3. **스타일**: 명확하고 전문적인 어조를 사용하세요.

        Response:
            ## 전체 요약
            (여기에 요약 문장 작성)

            ## 핵심 포인트
            - (포인트 1)
            - (포인트 2)
            ...

        [스크립트 전문]
        {text}
        """
    response = llm_client.call_llm(llm, system_msg, user_msg, mode)
    return response
