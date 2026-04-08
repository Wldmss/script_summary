# 1. 베이스 이미지: 순정 CentOS 7 사용
FROM centos:7

# 2. CentOS 7 EOL(지원종료) 대응: 저장소 URL을 vault로 변경 (필수)
RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-* && \
    sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-*

# 3. 필수 빌드 도구 설치 (Python 컴파일 및 C++ 라이브러리용)
RUN yum -y update && \
    yum -y groupinstall "Development Tools" && \
    yum -y install \
    openssl-devel \
    bzip2-devel \
    libffi-devel \
    xz-devel \
    wget \
    zlib-devel \
    sqlite-devel

# 4. Python 3.9.15 다운로드 및 컴파일 설치
WORKDIR /usr/src
RUN wget https://www.python.org/ftp/python/3.9.15/Python-3.9.15.tgz && \
    tar xzf Python-3.9.15.tgz && \
    cd Python-3.9.15 && \
    ./configure --enable-optimizations && \
    make altinstall && \
    rm -f /usr/src/Python-3.9.15.tgz

# 5. python3 명령어를 Python 3.9로 연결
RUN ln -s /usr/local/bin/python3.9 /usr/bin/python3 && \
    ln -s /usr/local/bin/pip3.9 /usr/bin/pip3

# 6. 작업 디렉토리 설정
WORKDIR /app

# 7. 환경 변수 설정
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# 확인용: 버전 출력
CMD ["python3", "--version"]