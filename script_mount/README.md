# llama_cpp_python 라이브러리 폴더 (삭제하면 안돼요)
/mount/Root
    - /llamaModel
        - libs : llama_cpp_python 설치에 필요한 의존성 라이브러리 모음
        - llama_cpp_python-0.2.90-cp39-cp39-manylinux2014_x86_64.manylinux_2_17_x86_64.whl : llama_cpp_python 라이브러리
        - llama-3.2-Korean-Bllossom-3B-Q8_0.gguf : 스크립트 요약 llm 모델
    - /scripts_summary : 스크립트 요약 파일 폴더
    

# llama_cpp_python 라이브러리 설치
```
cd /mount/Root/llamaModel

# 인터넷 끄는 옵션(--no-index)을 넣고 설치 시도
pip3 install --no-index --find-links=./libs ./llama_cpp_python-0.2.90-cp39-cp39-manylinux2014_x86_64.manylinux_2_17_x86_64.whl

# 1. 단순 import 테스트 (라이브러리 로딩 확인)
python3 -c 'from llama_cpp import Llama; print("성공! 완벽합니다!")'
```
