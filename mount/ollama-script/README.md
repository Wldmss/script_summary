-- Dockerfile 위치에서 실행

# 기존 requests와 urllib3를 지우고 구형 OpenSSL과 호환되는 버전으로 재설치

python3 -m pip install "urllib3<2" "requests<2.28.0" --break-system-packages

# 1. ai-python 컨테이너의 네트워크 확인

docker inspect ai-python -f '{{json .NetworkSettings.Networks}}'

# 2. ollama-server 컨테이너의 네트워크 확인

docker inspect ollama-server -f '{{json .NetworkSettings.Networks}}'

# 1. 이미지 빌드 (이름: ollama-eeve, 태그: v1) > 이미지 내려받기

cd ./mount/ollama-script

docker build -t ollama-eeve:v1 .

# 2. 빌드된 이미지를 tar 파일로 저장

docker save -o ollama-eeve-v1.tar ollama-eeve:v1

# container 생성

```운영
# 이미지 불러오기
docker load -i ollama-eeve-v1.tar

docker compose up -d

# 컨테이너 실행
docker run -d \
  --gpus all \
  --name ollama-server \
  --restart always \
  --network host \
  -e OLLAMA_KEEP_ALIVE=0 \
  -e OLLAMA_NUM_PARALLEL=1 \
  ollama-eeve:v1
```

```테스트용
docker compose -f docker-compose-test.yml up -d

docker run -d \
  --name ollama-test \
  -p 11434:11434 \
  -e OLLAMA_KEEP_ALIVE=-1 \
  ollama-eeve:v1
```

250415

- ollama 환경 세팅해보기
