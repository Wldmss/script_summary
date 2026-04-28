-- Dockerfile 위치에서 실행

# 기존 requests와 urllib3를 지우고 구형 OpenSSL과 호환되는 버전으로 재설치

python3 -m pip install "urllib3<2" "requests<2.28.0" --break-system-packages

# 1. ai-python 컨테이너의 네트워크 확인

docker inspect ai-python -f '{{json .NetworkSettings.Networks}}'

# 2. ollama-server 컨테이너의 네트워크 확인

docker inspect ollama-server -f '{{json .NetworkSettings.Networks}}'

---

# 1. 이미지 빌드 (이름: ollama-eeve, 태그: v1) > 이미지 내려받기

cd ./mount/ollama-script

docker build -t ollama-eeve:v1 .

# 운영 서버 아키텍처(amd64)에 맞춰 빌드

docker build --platform linux/amd64 -t ollama-eeve:v1-amd64 .

# 2. 빌드된 이미지를 tar 파일로 저장

docker save -o ollama-eeve-v1.tar ollama-eeve:v1

docker save -o ollama-eeve-v1-amd64.tar ollama-eeve:v1-amd64

---

# 1.5GB 분할

split -b 1500m ollama-eeve-v1-amd64.tar ollama-eeve-v1-amd64.tar.

git rm --cached ollama-eeve-v1-amd64.tar
git add ollama-eeve-v1-amd64.tar.\*
git commit -m "Add split tar files (under 2GB each)"
git push origin main

# fatal: '/Users/sqi/ktgenius/git/drive/.git/index.lock' 파일을 만들 수 없습니다: File exists.

rm -f .git/index.lock

# ollama 폴더 안의 모든 파일을 LFS로 관리하겠다는 설정

git lfs track "ollama/\*"

# 1. 폴더 추가

git add ollama/

# 2. .gitattributes 파일도 반드시 함께 추가 (LFS 설정 정보가 담겨 있음)

git add .gitattributes

# 3. 커밋

git commit -m "Add split ollama model files"

# tar 합치기

- window
  copy /b ollama-eeve-v1-amd64.tar.\* ollama-eeve-v1-amd64.tar

- mac
  cat ollama-eeve-v1-amd64.tar.\* > ollama-eeve-v1-amd64.tar

---

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
