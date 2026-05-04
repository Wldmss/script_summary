import os
import textwrap
import requests
from datetime import datetime

# ==========================================
# 설정값
# ==========================================
# 운영 환경(Host 모드)이라면 localhost, 테스트 환경(Bridge)이라면 컨테이너명 사용
# OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate") # 운영용 (host)
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://ollama-server:11434/api/generate")
MODEL_NAME = "eeve-expert:latest"

# 메모리 최적화를 위한 설정
# 10.8B 모델은 num_ctx가 너무 크면 메모리 부족(500 에러)이 발생하기 쉽습니다.
# 영상 스크립트 분량을 고려하여 4096~8192 사이를 추천합니다.
MAX_CONTEXT = 4096 
CHUNK_SIZE = 3000   # 안전하게 3000자 단위로 분할
CHUNK_OVERLAP = 300

SCRIPT_NAME = "20260116_BIZ_001"
SCRIPT_PATH = f"../scripts/{SCRIPT_NAME}.txt"

# ==========================================

def load_text(file_path):
    """파일 내용을 읽어옵니다."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def split_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """오버랩 포함 텍스트 분할 함수 (기존 로직 유지)"""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        if end >= text_len:
            chunks.append(text[start:])
            break

        last_newline = text.rfind('\n', start, end)
        if last_newline != -1:
            cut_point = last_newline + 1
        else:
            last_space = text.rfind(' ', start, end)
            if last_space != -1:
                cut_point = last_space + 1
            else:
                cut_point = end

        chunks.append(text[start:cut_point])
        start = cut_point - chunk_overlap
        if start < 0: start = 0
    
    return chunks

def call_ollama(system_msg, user_msg, mode):
    # Assistant가 답변을 시작할 형식을 미리 던져줍니다.
    if mode in ['final', 'single']:
        force_start = "1. **한 줄 요약:**"
        full_prompt = f"### System:\n{system_msg}\n\n### User:\n{user_msg}\n\n### Assistant:\n{force_start}"
    else:
        force_start = ""
        full_prompt = f"### System:\n{system_msg}\n\n### User:\n{user_msg}\n\n### Assistant:\n"
    
    payload = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,    # 최대로 낮춰서 변동성 제거
            "num_ctx": MAX_CONTEXT,
            "repeat_penalty": 1.2,
            "presence_penalty": 0.5, # 새로운 주제(줄바꿈)로 넘어갈 확률을 높임
            "stop": ["###", "User:", "Assistant:"]
        }
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=600)
        response.raise_for_status()
        result = response.json()['response'].strip()
        # 강제 시작 문구가 잘렸을 경우를 대비해 다시 붙여줍니다.
        return f"{force_start} {result}" if force_start else result
    except Exception as e:
        print(f"Ollama 호출 오류: {e}")
        return "오류 발생"

def summarize_chunk(text, mode='part'):
    """강의 성격에 맞는 요약 로직"""
    if mode == 'part':
        system_msg = "너는 복잡한 내용을 단문 위주의 리스트로 변환하는 요약 전문가야. 무조건 한 줄에 한 포인트만 작성해."
        user_msg = textwrap.dedent(f"""
            [지시사항]
            - 스크립트의 핵심 내용을 반드시 아래 [출력 형식]과 같이 한 줄씩 나누어 작성할 것.
            - 문단으로 뭉치지 말고, 각 포인트마다 줄바꿈을 할 것.
            - 문장 끝은 '~함', '~임'으로 짧게 끝낼 것.
            - 불필요한 서론(요약하자면 등)은 절대 금지.

            [출력 형식 예시]
            - 첫 번째 핵심 내용 포인트임
            - 두 번째 중요한 지식 포인트임
            - 세 번째 실습 관련 주의사항임

            [스크립트]
            {text}
            
            [결과]
            - 
        """).strip()

    elif mode == 'final':
        system_msg = "너는 온라인 강의 플랫폼의 베테랑 콘텐츠 에디터야."
        user_msg = f"""
            아래의 강의 요약 노트들을 바탕으로, 학습자들의 수강 욕구를 자극하는 **'강의 소개 요약본'**을 작성해줘.

            [구성]
            1. **한 줄 요약:** 강의의 핵심을 관통하는 매력적인 문구 (강조 기호 사용)
            2. **강의 상세 소개:** 이 강의가 무엇을 가르치는지, 어떤 가치를 제공하는지 친절한 문체로 서술 (3~4문장)
            3. **이런 분들께 추천해요:** 수강 대상자 3가지 포인트로 정리.

            [강의 요약 노트]
            {text}
        """

    else: # single (스크립트가 짧을 때 한 번에 처리)
        system_msg = "너는 교육용 영상의 매력적인 소개글을 쓰는 마케터이자 교육 전문가야."
        user_msg = f"""
            스크립트 전체를 읽고 학습자들의 수강 욕구를 자극하는 **'강의 소개 요약본'**을 작성해줘.

            [구성]
            1. **한 줄 요약:** 강의의 핵심을 관통하는 매력적인 문구 (강조 기호 사용)
            2. **강의 상세 소개:** 이 강의가 무엇을 가르치는지, 어떤 가치를 제공하는지 친절한 문체로 서술 (3~4문장)
            3. **이런 분들께 추천해요:** 수강 대상자 3가지 포인트로 정리.

            [스크립트 전문]
            {text}
        """

    return call_ollama(system_msg, user_msg, mode)

def main():
    print(f"[{datetime.now().time()}] 스케줄링 확인: Whisper 등 이전 프로세스 종료 대기...")
    # (실제 환경에서는 여기서 메모리 체크 로직을 넣을 수 있습니다)

    print(f"[{datetime.now().time()}] 스크립트 읽는 중...")
    try:
        script_content = load_text(SCRIPT_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    chunks = split_text(script_content)
    print(f"텍스트 길이: {len(script_content)}자 -> {len(chunks)}개 블록으로 분할")

    is_chunk = len(chunks) > 1
    summaries = []

    for i, chunk in enumerate(chunks):
        print(f"[{i+1}/{len(chunks)}] 부분 요약 생성 중 (Ollama)...")
        summary = summarize_chunk(chunk, mode='part' if is_chunk else 'single')
        summaries.append(summary)
        print(f"--- 결과 {i+1} ---\n{summary}\n")

    if is_chunk:
        print("최종 종합 요약 생성 중...")
        combined_text = "\n\n".join(summaries)
        final_summary = summarize_chunk(combined_text, mode='final')
    else:
        final_summary = summaries[0]

    print("\n" + "="*30)
    print(" [최종 요약 결과] ")
    print("="*30)
    print(final_summary)
    
    summary_file = f"../scripts/summary_{SCRIPT_NAME}_{datetime.now().strftime('%m%d_%H%M')}.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(final_summary)
    print(f"\n결과가 {summary_file} 에 저장되었습니다.")

if __name__ == "__main__":
    main()