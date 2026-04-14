source venv/bin/activate

python 3.9.15
centos linux 7 rhel fedora
mount 맞추기

/data/mount_docker:/mount/Root

---

# macOS 에서 docker 테스트 환경 만들기

# Docker 설치

```
% brew install orbstack
```

# Docker 실행

docker run -itd \
 --platform linux/amd64 \
 -v /Users/sqi/ktgenius/git/python/mount:/app \
 --name ai \
 centos7-py39 \
 /bin/bash

- ai-test : 패키징용
- ai : 테스트용

# 라이브러리 설치

```
cd /app

# 인터넷 끄는 옵션(--no-index)을 넣고 설치 시도
pip3 install --no-index --find-links=./libs ./llama_cpp_python-0.2.90-cp39-cp39-manylinux2014_x86_64.manylinux_2_17_x86_64.whl

# 1. 단순 import 테스트 (라이브러리 로딩 확인)
python3 -c 'from llama_cpp import Llama; print("성공! 완벽합니다!")'
```

# 라이브러리의 의존성 라이브러리 다운로드 (테스트)

```
pip3 download -d /app/libs ./llama-cpp-python-0.2.90.tar.gz

pip3 download -d /app/libs "scikit-build-core[pyproject]" cmake ninja
```

# 라이브러리 설치 (테스트)

```
pip3 install --no-index --find-links=/app/libs llama-cpp-python-0.2.90.tar.gz

-- gcc 9+ 버전으로 설치가 필요해 위에 명령어로는 안됨. -> C++ 컴파일이 필요해서 gcc 가 필요함
```

# gcc 9+ whl 만들기

```
    # 1. SCL 저장소 설치
    yum install -y centos-release-scl

    # 2. repo 주소 수정 (CentOS 7 EOL 대응)
        # 1. 모든 SCL 관련 repo 파일의 mirrorlist 주석 처리
        sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-SCLo-*.repo

        # 2. 모든 SCL 관련 repo 파일의 baseurl을 vault로 변경
        sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-SCLo-*.repo

        # 3. 꼬인 캐시 한 번 싹 비우기 (중요)
        yum clean all

    # 3. GCC 9 및 auditwheel(포장 도구) 설치
    yum install -y devtoolset-9-gcc devtoolset-9-gcc-c++ devtoolset-9-binutils
    pip3 install auditwheel patchelf

    # 1. GCC 9 활성화 (이 터미널에서만 유효)
    source /opt/rh/devtoolset-9/enable

    # 2. 빌드 도구 준비
    cd /app
    pip3 install "scikit-build-core[pyproject]" cmake ninja

    # 3. 설치(install)하지 말고 '파일 생성(wheel)'만 수행
    # 결과물은 /app/dist 폴더에 생깁니다.
    pip3 wheel ./llama_cpp_python-0.2.90.tar.gz -w /app/dist

    # 1. 수리된 파일을 저장할 폴더 생성
    mkdir /app/llama_wheel

    # 2. auditwheel로 수리 (repair)
    # dist 폴더에 있는 whl 파일을 찾아서 repair 명령을 내립니다.
    auditwheel repair /app/dist/llama_cpp_python-*.whl -w /app/llama_wheel


        # 컨테이너의 /app 폴더에서 실행
        cd /app

        # 소스 파일로 설치 진행 (GCC 9가 활성화된 상태여야 함!)
        # 이미 wheel을 만들었기 때문에 금방 설치됩니다.
        pip3 install ./llama-cpp-python-0.2.90.tar.gz

        # 1. libllama.so 파일 위치 찾기
        find /usr -name libllama.so
        # (보통 /usr/local/lib/python3.9/site-packages/llama_cpp/libllama.so 에 있습니다)

        # 2. 그 경로를 환경변수에 등록 (위 find 결과의 **폴더 경로**만 복사해서 넣으세요)
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.9/site-packages/llama_cpp


        find /usr -name libggml.so
        export LD_LIBRARY_PATH=/usr/local/lib/python3.9/site-packages/llama_cpp:$LD_LIBRARY_PATH
```

# requests 설치 파일 만들기

```
# mkdir -p /app/requests_libs

# requests와 함께, OpenSSL 1.0.2와 호환되는 urllib3 1.26.x 버전을 강제로 다운로드합니다.
pip3 download -d /app/requests_libs/ "requests" "urllib3<2.0"

# tar -cvzf requests.tar.gz /app/requests_libs  # 압축하는 경우
```

## requests 설치

```
# pip3 install --no-index --find-links=./requests_libs requests

# python3 -c "import requests; print('Requests 버전:', requests.__version__); print('성공: 라이브러리를 정상적으로 불러왔습니다')"
```

# github 100MB 업로드 안되는 경우

```
git config --global http.postBuffer 524288000
```
