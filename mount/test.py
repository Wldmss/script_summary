import os
import math
import textwrap
from datetime import datetime

# 사용할 모델 파일 경로
MODEL_PATH = "./llama-3.2-Korean-Bllossom-3B-Q8_0.gguf"

# 모델 컨텍스트 윈도우 크기 (CPU: 12288, GPU: 32768)
CTX_WINDOW = 12288

# 텍스트 분할 크기 (CTXT*0.7 정도 계산, CPU: 8000, GPU: 25000)
CHUNK_SIZE = 8000
CHUNK_OVERLAP = 500

# GPU 설정 (0: CPU만 사용 (느림), -1: VRAM이 허용하는 만큼 모든 레이어를 GPU에 올림 (권장))
N_GPU_LAYERS = 0

SCRIPT_NAME = "script3"
SCRIPT_PATH = f"./scripts/{SCRIPT_NAME}.txt"

# ==========================================

try:
    from llama_cpp import Llama
except ImportError:
    print("오류: 'llama-cpp-python' 라이브러리가 설치되지 않았습니다.")
    print("llama-cpp-python을 설치해주세요.")
    exit(1)

def load_text(file_path):
    """파일 내용을 읽어옵니다."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

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
        system_msg = "당신은 텍스트의 핵심 정보를 요약하는 AI입니다."
        raw_prompt = f"""
        ### [지시사항]
        아래 [스크립트] 내용을 읽고 핵심 내용만 간략히 요약하세요.
        
        1. **형식:** 반드시 '-(하이픈)'으로 시작하는 개조식(Bullet points)으로 작성할 것.
        2. **내용:** 해석이나 사족 없이 팩트만 건조하게 추출할 것.
        3. **금지:** "다음은 요약입니다" 같은 서론이나 인사를 절대 쓰지 말 것.

        ### [스크립트]
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
        system_msg = "당신은 영상 내용의 핵심 정보를 요약 정리하는 '전문가'입니다."
        user_msg = f"""
        다음은 교육 영상의 전체 스크립트입니다.
        학습자가 내용을 빠르게 파악할 수 있도록 **구조화된 요약본**을 작성해주세요.

        [지시사항]
        1. **구성:** 반드시 아래 두 파트로 나누어 작성할 것.
        - **전체 요약:** 영상의 주제와 결론을 3문장 이내의 자연스러운 줄글로 요약.
        - **핵심 포인트:** 중요 개념이나 배움 포인트를 3~5개의 글머리 기호(Bullet points)로 정리.
        2. **분량:** 전체 길이는 공백 포함 **300~500자**로 간결하게 작성할 것.
        3. **어조:** 군더더기 없이 지식을 전달하는 명확하고 전문적인 문체 사용. ('영상에서는', '말합니다' 등의 표현 제외)
        4. **용어:** 전문 용어는 변경하지 말고 그대로 사용할 것.

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

    # Chat Format 사용
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.1, # 사실적 요약을 위해 낮춤
        top_p=0.9,  # 뜬금없는 단어 방지
        max_tokens=2048 if mode in ['single', 'final'] else 600, # 최종 요약/단일 요약은 좀 더 길게 허용
        repeat_penalty=1.1,   # 같은 말 반복 방지
    )
    
    return response['choices'][0]['message']['content']

def main():
    print(f"[{datetime.now().time()}] 모델 로딩 중... ({MODEL_PATH})")
    
    if not os.path.exists(MODEL_PATH):
        print(f"주의: 모델 파일({MODEL_PATH})이 없습니다. 테스트 전 다운로드가 필요합니다.")
        # 코드가 멈추지 않도록 가짜 객체 처리 (실제 실행시는 제거)
        class MockLLM:
            def create_chat_completion(self, **kwargs):
                return {'choices': [{'message': {'content': '(모델 파일이 없어 생성된 더미 요약입니다.)'}}]}
        llm = MockLLM()
    else:
        llm = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=N_GPU_LAYERS, 
            n_ctx=CTX_WINDOW,
            verbose=False # 로그 줄이기
        )

    print(f"[{datetime.now().time()}] 스크립트 읽는 중...")
    script_content = load_text(SCRIPT_PATH)
    
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
        print("최종 종합 요약 생성 중...")
        combined_text = "\n\n".join(summaries)
        final_summary = summarize_chunk(llm, combined_text, mode='final')
    else:
        final_summary = summaries[0]

    print("\n" + "="*30)
    print(" [최종 요약 결과] ")
    print("="*30)
    print(final_summary)
    
    # 결과 파일 저장
    summary_file = f"./scripts/summary_{SCRIPT_NAME}_{datetime.now().strftime('%m%d_%H%M')}.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(final_summary)
    print(f"\n결과가 {summary_file} 에 저장되었습니다.")

if __name__ == "__main__":
    main()