import os
import json
import textwrap
import re
import argparse
from datetime import datetime
from typing import List, Any, Dict

# python3 quiz2.py --model_type qwen --use_summary

DEFAULT_MODEL_TYPE = "qwen"
MODEL_MAP = {
    "qwen": "./Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    "exaone": "./EXAONE-3.0-7.8B-Instruct-Q4_K_M.gguf",
    "llama-8b": "./llama-3-Korean-Bllossom-8B.Q4_K_M.gguf",
    "llama-3b": "./llama-3.2-Korean-Bllossom-3B-Q8_0.gguf"
}

# 스크립트 원본 경로
DEFAULT_SCRIPT_NAME = "20260116_BIZ_001"

# 스크립트 요약본
SUMMARY_PATH = "./scripts/summary_qwen_20260116_BIZ_001_0120_0730.txt"

# 모델 설정
CTX_WINDOW = 8192
N_GPU_LAYERS = -1 

# 텍스트 분할 크기 (CTXT*0.7 정도 계산, CPU: 8000, GPU: 25000)
CHUNK_SIZE = 8000
CHUNK_OVERLAP = 500

try:
    from llama_cpp import Llama
except ImportError:
    print("오류: 'llama-cpp-python' 라이브러리가 설치되지 않았습니다.")
    exit(1)

def load_text(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def save_file(file_path: str, content: Any):
    """
    내용(content)의 타입에 따라 텍스트 또는 JSON으로 파일을 저장합니다.
    """
    try:
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            # 내용이 리스트나 딕셔너리면 JSON 형식으로 저장
            if isinstance(content, (list, dict)):
                json.dump(content, f, ensure_ascii=False, indent=2)
                print(f"\n[JSON] 파일 저장 완료: {file_path}")
                if isinstance(content, list):
                    print(f" - 항목 수: {len(content)}")
            
            # 내용이 문자열이면 일반 텍스트로 저장
            else:
                f.write(str(content))
                print(f"\n[TEXT] 파일 저장 완료: {file_path}")
                
    except Exception as e:
        print(f"파일 저장 실패: {e}")

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

def extract_json(text: str) -> Any:
    """텍스트에서 JSON 부분만 스마트하게 추출"""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 리스트 [] 추출 시도
    match = re.search(r'(\[.*\])', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
            
    # 객체 {} 추출 시도
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    return None

def generate_topics(llm: Llama, summary: str, count: int = 5) -> List[str]:
    """
    [1단계] 요약본을 분석하여 시험에 나올만한 핵심 키워드/주제 리스트를 추출합니다.
    """
    print(f"\n[1단계] AI가 핵심 주제 {count}가지를 추출하는 중...")
    
    # system_msg = "당신은 교육 콘텐츠 분석가입니다. 텍스트에서 핵심 주제를 키워드 중심으로 추출하세요."
    # user_msg = f"""
    # 아래 [요약본]을 분석하여, 객관식 시험 문제로 출제하기 좋은 **서로 다른 핵심 주제 {count}가지**를 선정하여 JSON 리스트로 반환하시오.

    # [제약 사항]
    # 1. 주제는 명사형이나 짧은 구문으로 작성할 것. (예: "감마의 가입 방법", "무료와 유료의 차이")
    # 2. 주제끼리 내용이 겹치지 않게 전체 내용을 고루 분배할 것.
    # 3. 오직 JSON List 형식만 출력할 것.

    # [출력 예시]
    # ["주제1", "주제2", "주제3", "주제4", "주제5"]

    # [요약본]
    # {summary}
    # """

    system_msg = "당신은 시험 출제 범위를 선정하는 분석가입니다."
    user_msg = f"""
    아래 [요약본]을 분석하여, 객관식 문제로 출제하기 좋은 **서로 완전히 다른 핵심 주제 {count}가지**를 선정하여 JSON 리스트로 반환하시오.

    [제약 사항 - 필독]
    1. **중복 금지**: 비슷한 주제는 절대 피할 것. (예: 'AI의 역할', 'AI의 중요성' -> 중복임. 하나만 선택)
    2. **구체성**: 포괄적인 단어보다 구체적인 키워드를 선호함. (예: '사회' (X) -> 'OTT 구독 습관' (O))
    3. **형식**: 오직 JSON List 형식만 출력할 것.

    [나쁜 예시]
    ["AI", "인공지능", "AI의 기술", "미래", "전망"] (내용이 다 겹침)

    [좋은 예시]
    ["메타버스 검색량 추이", "OTT 구독률 통계", "카운터컬쳐 트렌드", "결혼정보회사 매출 변화", "메타인지의 필요성"]

    [요약본]
    {summary}
    """

    response = llm.create_chat_completion(
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        temperature=0.1,
        max_tokens=512,
    )
    
    result = extract_json(response['choices'][0]['message']['content'])
    
    # 결과 검증
    if isinstance(result, list) and len(result) > 0:
        # 혹시 개수가 부족하면 나머지는 무시하거나 리스트만 반환
        return result[:count]
    else:
        print("  -> 주제 추출 실패, 기본 주제로 대체합니다.")
        return ["", "", "", "", ""]

def generate_single_quiz(llm: Llama, summary: str, target_topic: str) -> Dict:
    """
    [2단계] 특정 주제(target_topic)에 대해 짧고 간결한 문제 생성
    """
    
    system_msg = "당신은 간결하고 명확한 문제를 만드는 시험 출제 위원입니다."
    
    # user_msg = f"""
    # [요약본]을 참고하여 {"**'{target_topic}'**에 관한" if target_topic == "" else ""} 객관식 문제 1개를 만드십시오.
    
    # [필수 제약 조건 - 길이 제한]
    # 1. **질문**: 50자 이내로 핵심만 물어보는 의문형 일 것. (예: "감마 플랫폼의 기본 제공 크레딧 양은?")
    # 2. **보기**: **20자 이내**로 아주 짧게 작성할 것. 서술형 문장 금지. (예: "500 크레딧")
    # 3. **정답**: 논란의 여지가 없는 확실한 사실(Fact)일 것.

    # [출력 포맷 JSON]
    # {{
    #     "question": "짧은 질문 텍스트",
    #     "options": ["짧은 보기1", "짧은 보기2", "짧은 보기3", "짧은 보기4"],
    #     "answer": 1, 
    #     "rationale": "짧은 해설"
    # }}
    # * answer는 1, 2, 3, 4 중 하나.

    # [주제]: {target_topic}
    # [요약본]: {summary}
    # """

    user_msg = f"""
    아래 [요약본]을 참고하여, 반드시 **'{target_topic}'**에 관한 객관식 문제 1개를 만드십시오.

    [제약 조건 - 엄격 준수]
    1. **질문(Question)**: 
       - 반드시 **물음표(?)**로 끝나는 의문문이어야 함.
       - "~인가?", "~은 무엇인가?", "~으로 알맞은 것은?" 형태 사용.
       - (나쁜 예: "AI의 중요성") -> (좋은 예: "AI의 중요성이 강조된 시기는 언제인가?")
       
    2. **보기(Options)**: 
       - **15자 이내**의 단답형 또는 명사형으로 작성할 것.
       - 문장으로 길게 쓰지 말 것. (설명은 해설에 넣으세요)
       - (나쁜 예: "사람들이 외로움을 느껴서 결혼정보회사를 많이 찾게 되었다.")
       - (좋은 예: "외로움 증가")
       
    3. **정답(Answer)**: 
       - 요약본에 있는 내용에 근거할 것. (지어내지 말 것)
       
    4. **형식**: JSON 포맷 준수.

    [출력 예시]
    {{
        "question": "최근 결혼정보회사의 매출이 급증한 주된 원인은 무엇인가?",
        "options": ["경제 호황", "외로움 증가", "AI 기술 발전", "정부 지원"],
        "answer": 2, 
        "rationale": "사람들의 외로움이 심해지면서 역설적으로 결혼정보회사의 매출이 급증함"
    }}

    [주제]: {target_topic}
    [요약본]: {summary}
    """

    response = llm.create_chat_completion(
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        temperature=0.2, 
        max_tokens=512,
        response_format={"type": "json_object"} 
    )
    
    return extract_json(response['choices'][0]['message']['content'])

def main(args):
    print(f"[{datetime.now().time()}] 프로세스 시작...")

    model_type = args.model_type
    use_summary = args.use_summary
    script_name = args.script_name
    script_path = f"./scripts/{script_name}.txt"

    if model_type not in MODEL_MAP:
        print(f"오류: 지원하지 않는 모델 타입입니다: {model_type}")
        print(f"가능한 모델: {list(MODEL_MAP.keys())}")
        return

    model_path = MODEL_MAP[model_type]

    print(f" - Model Type: {model_type}")
    print(f" - Model Path: {model_path}")
    print(f" - Script: {script_path}")
    print(f" - Use Summary: {use_summary} | {SUMMARY_PATH if use_summary else ''}")

    summary_title = f"summary_{model_type}_{script_name}_{datetime.now().strftime('%m%d_%H%M')}.txt"
    quiz_title = f"quiz_{model_type}_{script_name}_{datetime.now().strftime('%m%d_%H%M')}.json"
    
    if not os.path.exists(model_path):
        print(f"오류: 모델 파일 확인 필요")
        return

    llm = Llama(
        model_path=model_path,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=CTX_WINDOW,
        verbose=False
    )

    if use_summary:
        if not os.path.exists(SUMMARY_PATH):
            print("요약 파일이 없습니다.")
            return
        
        final_summary = load_text(SUMMARY_PATH)
        print(f"요약본 로드 완료 ({len(final_summary)}자)")
    else:
        print(f"[{datetime.now().time()}] 스크립트 읽는 중...")
        script_content = load_text(script_path)
        
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
            final_summary = "\n\n".join(summaries)
        else:
            final_summary = summaries[0]

        print("\n" + "="*30)
        print(" [최종 요약 결과] ")
        print("="*30)
        print(final_summary)
        
        # 결과 파일 저장
        save_file(f"./scripts/{summary_title}", final_summary)

    # ==========================================
    # 1. AI가 스스로 주제 추출 (Dynamic Topic Extraction)
    # ==========================================
    target_count = 5
    topics = generate_topics(llm, final_summary, target_count)
    print(f"  -> 추출된 주제: {topics}")

    # ==========================================
    # 2. 주제별 문제 생성
    # ==========================================
    generated_questions = []
    print(f"\n[2단계] 주제별 퀴즈 생성 시작...")

    for i, topic in enumerate(topics):
        print(f"  -> [{i+1}/{len(topics)}] 주제: '{topic}' 생성 중...")
        
        for attempt in range(2): # 실패 시 1회 재시도
            try:
                quiz_data = generate_single_quiz(llm, final_summary, topic)
                
                # 데이터 유효성 검사
                if quiz_data and "question" in quiz_data and len(quiz_data.get("options", [])) == 4:
                    quiz_data["question_ser"] = i + 1
                    quiz_data["related_topic"] = topic # (선택) 어떤 주제로 뽑았는지 저장
                    generated_questions.append(quiz_data)
                    print(f"    성공: {quiz_data['question']}")
                    break
                else:
                    print(f"    실패 (형식 불일치)")
            except Exception as e:
                print(f"    에러: {e}")

    # 저장
    if generated_questions:
        save_file(f"./scripts/{quiz_title}", generated_questions)
        print(f"총 문제 수: {len(generated_questions)}")
    else:
        print("생성된 문제가 없습니다.")

if __name__ == "__main__":
    # 인자 파서 설정
    parser = argparse.ArgumentParser(description="AI Quiz Generator")

    # 1. 모델 타입 설정 (--model_type qwen)
    parser.add_argument(
        "--model_type", 
        type=str, 
        default=DEFAULT_MODEL_TYPE, 
        choices=MODEL_MAP.keys(),
        help="사용할 LLM 모델 타입 (qwen, exaone, llama-8b, llama-3b)"
    )

    # 2. 스크립트 이름 설정 (--script_name 파일명)
    parser.add_argument(
        "--script_name", 
        type=str, 
        default=DEFAULT_SCRIPT_NAME,
        help="스크립트 파일 이름 (확장자 제외)"
    )

    # 3. 요약본 사용 여부 (--use_summary)
    # 이 옵션을 넣으면 True, 안 넣으면 False가 됩니다.
    parser.add_argument(
        "--use_summary", 
        action="store_true", 
        help="이미 생성된 요약본을 사용할지 여부 (기본값: False)"
    )

    args = parser.parse_args()
    main(args)