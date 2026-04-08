import os
import math
import json
import re
import textwrap
from datetime import datetime
from typing import List, Dict, Any

# 설정
MODEL_PATH = "./llama-3.2-Korean-Bllossom-3B-Q8_0.gguf"
SCRIPT_PATH = "./scripts/script4.txt" # 사용자가 직접 지정하는 스크립트 파일 경로

# 모델 및 청크 설정
CTX_WINDOW = 12288
CHUNK_SIZE = 8000 # 한 번에 처리할 텍스트 크기
CHUNK_OVERLAP = 500 # 청크 간 겹치는 구간 (문맥 유지)
N_GPU_LAYERS = -1 

try:
    from llama_cpp import Llama
except ImportError:
    print("오류: 'llama-cpp-python' 라이브러리가 설치되지 않았습니다.")
    print("pip install llama-cpp-python 명령어로 설치해주세요.")
    exit(1)

def load_text(file_path: str) -> str:
    """파일 내용을 읽어옵니다."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    텍스트를 지정된 크기로 분할합니다 (오버랩 포함).
    """
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
            cut_point = last_space + 1 if last_space != -1 else end

        chunks.append(text[start:cut_point])
        start = cut_point - chunk_overlap
        if start < 0: start = 0
    
    return chunks

def extract_json(text: str) -> Any:
    """
    텍스트에서 JSON 블록을 추출하여 파이썬 객체(list 또는 dict)로 변환합니다.
    중첩 인코딩(문자열 안에 JSON) 케이스도 대응합니다.
    """
    text = text.strip()
    
    def try_parse(s: str) -> Any:
        try:
            val = json.loads(s)
            # 만약 파싱 결과가 문자열이면, 그 문자열이 다시 JSON일 수 있음 (중첩 인코딩)
            if isinstance(val, str):
                try:
                    inner_val = json.loads(val)
                    if isinstance(inner_val, (list, dict)):
                        return inner_val
                except:
                    pass
            return val
        except json.JSONDecodeError:
            return None

    # 1. 전체 텍스트 파싱 시도
    result = try_parse(text)
    if isinstance(result, (list, dict)):
        return result

    # 2. 마크다운 코드 블록 추출 시도
    match = re.search(r'```json\s*(.*?)```', text, re.DOTALL)
    if match:
        result = try_parse(match.group(1).strip())
        if isinstance(result, (list, dict)):
            return result

    # 3. 리스트([]) 추출 시도
    list_match = re.search(r'(\[.*\])', text, re.DOTALL)
    if list_match:
        result = try_parse(list_match.group(1).strip())
        if isinstance(result, (list, dict)):
            return result

    # 4. 중괄호({}) 추출 시도
    dict_match = re.search(r'(\{.*\})', text, re.DOTALL)
    if dict_match:
        result = try_parse(dict_match.group(1).strip())
        if isinstance(result, (list, dict)):
            return result
            
    return None

def generate_quiz_chunk(llm: Llama, text: str, num_questions: int, chunk_index: int, total_chunks: int) -> str:
    """
    특정 텍스트 청크에서 지정된 개수만큼의 퀴즈를 JSON 형식으로 생성합니다.
    """
    system_msg = "당신은 교육 영상의 내용을 바탕으로 '객관식 시험 문제'를 출제하는 엄격한 시험 출제 위원입니다. 정답은 1개이며, 오답은 매력적이어야 합니다."
    
    user_msg = f"""
    # Role (역할)
    당신은 IT 및 직무 교육 분야의 전문 콘텐츠 제작자이자 평가 문항 개발 전문가입니다. 주어진 교육 영상의 스크립트를 완벽하게 이해하고, 학습자의 이해도를 점검할 수 있는 고품질의 객관식 퀴즈를 출제하는 것이 당신의 임무입니다.

    # Task (임무)
    아래 [스크립트 파트 {chunk_index+1}/{total_chunks}]를 읽고, **정확히 {num_questions}개의 서로 다른 객관식 퀴즈**를 JSON 리스트 형태로 반환하십시오.

    # Constraints (제약 조건 - 반드시 지킬 것)
    1. **문제의 질**: 강사의 농담, 인사말, 사담 등 불필요한 내용은 철저히 배제하고, 오직 '학습 핵심 개념'과 '중요 지식'에 대해서만 문제를 출제하십시오.
    2. **문항 수**: 정확히 {num_questions} 개의 문제를 생성하십시오.
    3. **유형의 다양성**: 단순히 맞는 것을 고르는 문제뿐만 아니라, "다음 중 틀린 것은?", "가장 적절하지 않은 것은?"과 같은 부정형 질문도 적절히 섞어서 출제하십시오.
    4. **보기(options) 구성**:
    - 모든 문제는 4지선다(보기 4개)여야 합니다.
    - 보기 4개는 서로 중복되거나 동일한 내용이 있어서는 안 됩니다.
    - 정답은 다른 오답과 대비해 명확하게 1개여야 합니다.
    5. **난이도**: 스크립트를 제대로 읽거나 듣지 않으면 풀 수 없도록, 지나치게 상식적인 내용은 피하고 스크립트 기반의 구체적인 사실을 물어보십시오.
    6. **출력 형식**: 반드시 아래 정의된 [Output JSON Schema]를 따르십시오. 마크다운(```json) 태그 외에 불필요한 서술은 하지 마십시오.
    7. "answer"는 정답 보기의 인덱스 + 1(1부터 시작: 1, 2, 3, 4 중 하나)인 정수여야 합니다.

    # Output JSON Schema (출력 포맷)
    ```json
    {{
        "question_ser": 1,
        "question": "문제 지문 (예: 다음 중 파이썬의 특징으로 옳지 않은 것은?)",
        "options": [
        "보기 1번 텍스트",
        "보기 2번 텍스트",
        "보기 3번 텍스트",
        "보기 4번 텍스트"
        ],
        "answer": 1,
        "rationale": "정답에 대한 간략한 해설 (스크립트 근거)"
    }},
    ...
    ```

    [스크립트 파트]
    {text}
    """

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.1, # 포맷 준수 및 정확도 향상
        max_tokens=2048, 
        response_format={"type": "json_object"} 
    )
    
    return response['choices'][0]['message']['content']

def main():
    print(f"[{datetime.now().time()}] 초기화 중...")
    
    if not os.path.exists(SCRIPT_PATH):
        print(f"오류: 스크립트 파일이 없습니다: {SCRIPT_PATH}")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"오류: 모델 파일이 없습니다: {MODEL_PATH}")
        return

    try:
        full_text = load_text(SCRIPT_PATH)
        print(f"스크립트 로드 완료: {len(full_text)}자")
    except Exception as e:
        print(f"스크립트 로드 실패: {e}")
        return

    chunks = split_text(full_text)
    print(f"텍스트 분할: 총 {len(chunks)}개 블록")

    print("모델 로딩 중...")
    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=CTX_WINDOW,
            verbose=False
        )
    except Exception as e:
        print(f"모델 로드 에러: {e}")
        return

    print("퀴즈 생성 시작 (JSON 모드)...")
    
    total_target_questions = 10
    all_questions_data = []
    
    base_count = total_target_questions // len(chunks)
    remainder = total_target_questions % len(chunks)
    
    for i, chunk in enumerate(chunks):
        count = base_count + (1 if i < remainder else 0)
        if count == 0: continue
            
        print(f"[{i+1}/{len(chunks)}] 청크 처리 중 (목표: {count}문제)...")
        
        try:
            raw_response = generate_quiz_chunk(llm, chunk, count, i, len(chunks))
            print(f"  -> {raw_response}")

            parsed_data = extract_json(raw_response)
            
            if parsed_data:
                # If the model returns a single question dict instead of a list,
                # wrap it in a list so downstream code can treat it uniformly.
                if isinstance(parsed_data, dict) and not isinstance(parsed_data, list):
                    # Handle case where a dict represents a single question
                    # or contains a "questions" key.
                    if "questions" in parsed_data:
                        parsed_data = parsed_data["questions"]
                    else:
                        # Assume the dict itself is a single question
                        parsed_data = [parsed_data]

                if isinstance(parsed_data, list):
                    all_questions_data.extend(parsed_data)
                    print(f"  -> {len(parsed_data)}문제 파싱 성공")
                else:
                    print(f"  -> 파싱 성공했으나 리스트 형식이 아님: {type(parsed_data)}")
            else:
                print(f"  -> JSON 파싱 실패. 원본 응답:\\n{raw_response[:100]}...")
                return
                
        except Exception as e:
            print(f"  -> 생성 또는 파싱 중 에러: {e}")

    # 결과 저장
    if not all_questions_data:
        print("\n생성된 퀴즈가 없습니다.")
        return

    # 전역 순번으로 question_ser 재설정 (1부터 순차적으로)
    for i, q in enumerate(all_questions_data, start=1):
        if isinstance(q, dict):
            q["question_ser"] = i

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"quiz_json_{os.path.basename(SCRIPT_PATH).split('.')[0]}_{timestamp}.json"
    output_dir = os.path.dirname(SCRIPT_PATH)
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_questions_data, f, ensure_ascii=False, indent=2)
        print(f"\n퀴즈 파일 저장 완료 (JSON): {output_path}")
        print(f"총 문제 수: {len(all_questions_data)}")
    except Exception as e:
        print(f"파일 저장 실패: {e}")

if __name__ == "__main__":
    main()
