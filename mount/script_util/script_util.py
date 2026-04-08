import os
import json
import textwrap
import re
import argparse
from datetime import datetime
from typing import List, Any, Dict
import llm_client
import summary_util
import quiz_util

# python3 script_util.py --model_type qwen --use_summary

# 스크립트 원본 경로
DEFAULT_SCRIPT_NAME = "20260116_BIZ_001"

# 스크립트 요약본
SUMMARY_PATH = "./scripts/summary_qwen_20260116_BIZ_001_0120_0730.txt"

# 텍스트 분할 크기 (CTXT*0.7 정도 계산, CPU: 8000, GPU: 25000)
CHUNK_SIZE = 8000
CHUNK_OVERLAP = 500

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

    llm = load_llm(model_type)

    # 스크립트 요약 생성
    if use_summary: 
        if not os.path.exists(summary_path):
            raise ValueError(f"요약 파일이 없습니다: {summary_path}")
        
        summary_text = load_text(summary_path)
        print(f"요약본 로드 완료 ({len(summary_text)}자)")

    else:
        print(f"[{datetime.now().time()}] 스크립트 읽는 중...")
        script_content = load_text(script_path)
        summary_text = summary_util.summarize_script(llm, script_content, summary_path, summary_title)
    
    # 퀴즈 생성
    quiz_util.generate_quiz(llm, summary_text, quiz_title)

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