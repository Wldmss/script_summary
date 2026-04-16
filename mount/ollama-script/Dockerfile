# 1. Ollama 베이스 이미지
FROM ollama/ollama:latest

# 2. 빌드 과정에서 모델 미리 내려받기
# 모델명을 확인하신 'anpigon/eeve-korean-10.8b:latest'로 교체했습니다.
RUN nohup bash -c "ollama serve &" && \
    sleep 10 && \
    # 확인하신 정확한 모델 경로로 pull 시도
    ollama pull anpigon/eeve-korean-10.8b:latest && \
    # 3. 커스텀 모델 생성 (기반 모델을 위와 동일하게 설정)
    echo 'FROM anpigon/eeve-korean-10.8b:latest\n\
PARAMETER num_ctx 16384\n\
PARAMETER temperature 0.3\n\
SYSTEM "너는 한국어 영상 콘텐츠 분석 전문가야. 모든 답변은 한국어로 작성해."' > eeve_modelfile && \
    ollama create eeve-expert -f eeve_modelfile && \
    pkill ollama

# 4. 환경 설정
ENV OLLAMA_HOST=0.0.0.0
EXPOSE 11434

ENTRYPOINT ["/bin/ollama", "serve"]