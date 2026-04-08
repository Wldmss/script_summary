import os
import json
import textwrap
import re
import argparse
from datetime import datetime
from typing import List, Any, Dict

DEFAULT_MODEL_TYPE = "qwen"
MODEL_MAP = {
    "qwen": "./Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    "exaone": "./EXAONE-3.0-7.8B-Instruct-Q4_K_M.gguf",
    "llama-8b": "./llama-3-Korean-Bllossom-8B.Q4_K_M.gguf",
    "llama-3b": "./llama-3.2-Korean-Bllossom-3B-Q8_0.gguf"
}

# 모델 설정
CTX_WINDOW = 8192
N_GPU_LAYERS = -1

try:
    from llama_cpp import Llama
except ImportError:
    print("오류: 'llama-cpp-python' 라이브러리가 설치되지 않았습니다.")
    exit(1)

def load_llm(model_type: str) -> Llama:
    model_path = MODEL_MAP.get(model_type)
    if not model_path:
        raise ValueError(f"알 수 없는 모델 타입: {model_type}")
    
    llm = Llama(
        model_path=model_path,
        n_ctx=CTX_WINDOW,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False
    )
    return llm

def call_llm(llm: Llama, system_msg: str, user_msg: str, mode: str = 'part') -> str:
    # Chat Format 사용
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    if mode == 'json':
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.2, 
            max_tokens=512,
            response_format={"type": "json_object"} 
        )
    elif mode == 'topic':
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=512,
        )
    else:   # part single final
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.1, # 사실적 요약을 위해 낮춤
            top_p=0.9,  # 뜬금없는 단어 방지
            max_tokens=2048 if mode in ['single', 'final'] else 600, # 최종 요약/단일 요약은 좀 더 길게 허용
            repeat_penalty=1.1,   # 같은 말 반복 방지
        )

    return response['choices'][0]['text'].strip()