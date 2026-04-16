import os
import json
import textwrap
import requests  # pip install requests (2.27.1 버전 권장)

# 설정값 (기존 설정 유지)
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 200
# OLLAMA_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate") # 운영용 (host)
OLLAMA_URL = os.getenv("OLLAMA_API_URL", "http://ollama-server:11434/api/generate") # 테스트용 (bridge)


def summarize_script(script_content: str, summary_title: str):
    """
    Ollama를 사용하여 스크립트를 요약하는 메인 함수
    """
    # 1. 텍스트 분할 (기존 split_text 함수 그대로 사용 가능)
    chunks = split_text(script_content)
    print(f"텍스트 길이: {len(script_content)}자 -> {len(chunks)}개 블록으로 분할")

    is_chunk = len(chunks) > 1
    summaries = []

    # 2. 부분 요약 (Map 단계)
    for i, chunk in enumerate(chunks):
        print(f"[{i+1}/{len(chunks)}] 요약 생성 중...")
        # 'eeve-expert'는 Dockerfile에서 생성한 모델명입니다.
        summary = summarize_chunk(chunk, mode='part' if is_chunk else 'single')
        summaries.append(summary)
        print(f"--- 부분 요약 {i+1} ---\n{summary}\n")

    # 3. 최종 요약 (Reduce 단계)
    if is_chunk:
        summary_text = "\n\n".join(summaries)
        final_summary = summarize_chunk(summary_text, mode='final')
    else:
        summary_text = summaries[0]
        final_summary = summary_text

    print("\n" + "="*30)
    print(" [최종 요약 결과] ")
    print("="*30)
    print(final_summary)
    
    # 결과 파일 저장 로직 (기존 함수가 있다고 가정)
    # save_file(f"./scripts/final_{summary_title}", final_summary)
    
    return final_summary

def summarize_chunk(text, mode='part'):
    """
    Ollama API를 호출하여 실제로 AI 응답을 받아오는 함수
    """
    # 프롬프트 구성 (기존 작성하신 내용 유지)
    if mode == 'part':
        system_prompt = "당신은 강의 내용을 논리적으로 분석하여 정리하는 '수석 연구원'입니다."
        user_prompt = f"""
        [지시사항]
        제공된 [스크립트]를 읽고, 핵심 내용을 **상세한 노트 필기 형식**으로 정리하세요.
        단순한 사실 나열이 아니라, **'무엇이(What)', '왜(Why)', '어떻게(How)', '결과(Result)'**의 논리적 흐름이 보이도록 작성해야 합니다.
        1. **형식**: 반드시 `-(하이픈) **핵심키워드**: 상세 설명` 형식을 지키십시오.
        2. **내용 깊이**: 각 항목은 2~3문장으로 구체적으로 서술하십시오.
        [스크립트]
        {text}
        ### [요약 결과]
        -"""
    
    elif mode == 'final':
        system_prompt = "당신은 복잡한 정보를 통찰력 있게 정리하여 보고서를 작성하는 '전문 에디터'입니다."
        user_prompt = f"""
        다음은 영상의 핵심 내용을 정리한 노트들입니다. 이를 바탕으로 학습자를 위한 **구조화된 요약본**을 작성해주세요.
        [지시사항]
        1. **전체 요약:** 영상의 주제와 결론을 자연스러운 문장으로 서술하세요.
        2. **분량:** 전체 길이는 공백 포함 300~500자 정도로 맞추세요.
        [부분 요약본 내용]
        {text}
        """
    
    else:  # single
        system_prompt = "당신은 복잡한 정보를 통찰력 있게 정리하여 교육 자료를 만드는 '전문 에디터'입니다."
        user_prompt = f"""
        다음은 교육 영상의 전체 스크립트입니다. 이를 종합하여 학습자를 위한 **완결성 있는 최종 요약보고서**를 작성해 주세요.
        [지시사항]
        1. **## 전체 요약**: 주제, 배경, 결론을 아우르는 3~4문장의 줄글 작성.
        2. **## 핵심 포인트**: 중요한 개념, 수치 등을 5~7개의 개조식으로 정리.
        [스크립트 전문]
        {text}
        """

    # Ollama API 호출
    # Dockerfile에서 'eeve-expert'라는 이름으로 모델을 생성했으므로 해당 이름을 사용합니다.
    payload = {
        "model": "eeve-expert",
        "prompt": f"{system_prompt}\n\n{user_prompt}",
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_ctx": 8192
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300) # 5분 타임아웃
        response.raise_for_status()
        result = response.json()
        return result.get('response', '').strip()
    except Exception as e:
        print(f"API 호출 중 오류 발생: {e}")
        return "요약 생성 실패"


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
    