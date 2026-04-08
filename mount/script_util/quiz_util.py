import os
import json
import textwrap
import re
import argparse
from datetime import datetime
from typing import List, Any, Dict
import llm_client

def generate_quiz(llm, summary: str, quiz_title: str):
    
    # Quiz 주제 추출
    target_count = 5
    topics = generate_topics(llm, summary, target_count)
    print(f"  -> 추출된 주제: {topics}")

    # 주제별 퀴즈 생성
    generated_questions = []
    print(f"\n퀴즈 생성 시작...")

    for i, topic in enumerate(topics):
        print(f"  -> [{i+1}/{len(topics)}] 주제: '{topic}' 생성 중...")
        
        for attempt in range(2): # 실패 시 1회 재시도
            try:
                quiz_data = generate_single_quiz(llm, summary, topic)
                
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

    response = llm_client.call_llm(llm, system_msg, user_msg, mode='topic')
    result = extract_json(response)
    
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
        "rationale": "요약본에 따르면 사람들의 외로움이 심해지면서 역설적으로 결혼정보회사의 매출이 급증했다고 언급됨."
    }}

    [주제]: {target_topic}
    [요약본]: {summary}
    """

    response = llm_client.call_llm(llm, system_msg, user_msg, mode='json')
    return extract_json(response)

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