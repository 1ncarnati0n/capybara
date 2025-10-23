#!/bin/bash

# 스크립트가 있는 디렉토리로 이동
cd "$(dirname "$0")"

echo "================================"
echo "AI 번역기 카피바라 (Capybara)"
echo "vLLM 기반 고속 번역"
echo "================================"
echo ""

# Conda 환경 체크
if ! command -v conda &> /dev/null
then
    echo "⚠️  Conda가 설치되어 있지 않습니다."
    echo "vLLM은 Python 3.11 환경이 필요하므로 Conda 사용을 권장합니다."
    echo ""
    echo "Conda 설치 후 다음 명령어를 실행하세요:"
    echo "  conda create -n capybara python=3.11 -y"
    echo "  conda activate capybara"
    echo "  pip install -r requirements.txt"
    echo ""
    read -p "그래도 계속하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]
    then
        exit 1
    fi
fi

# Conda 환경 활성화 체크
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "⚠️  Conda 환경이 활성화되어 있지 않습니다."
    echo ""
    echo "다음 명령어로 환경을 활성화하세요:"
    echo "  conda create -n capybara python=3.11 -y  # 처음 한 번만"
    echo "  conda activate capybara"
    echo "  pip install -r requirements.txt  # 처음 한 번만"
    echo ""
    read -p "그래도 계속하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]
    then
        exit 1
    fi
else
    echo "✅ Conda 환경: $CONDA_DEFAULT_ENV"
fi

# Python 버전 체크
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python 버전: $PYTHON_VERSION"

# vLLM 설치 체크
if ! python -c "import vllm" &> /dev/null; then
    echo ""
    echo "⚠️  vLLM이 설치되어 있지 않습니다."
    echo "다음 명령어로 설치하세요:"
    echo "  pip install -r requirements.txt"
    echo ""
    read -p "지금 설치하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        echo "패키지 설치 중..."
        pip install -r requirements.txt

        if [ $? -ne 0 ]; then
            echo ""
            echo "❌ 설치에 실패했습니다."
            echo "다음을 확인하세요:"
            echo "  1. CUDA 11.8 이상 설치되어 있는지"
            echo "  2. Python 3.11 사용 중인지"
            echo "  3. pip가 최신 버전인지 (pip install --upgrade pip)"
            exit 1
        fi

        echo "✅ 설치 완료!"
    else
        exit 1
    fi
else
    echo "✅ vLLM 설치됨"
fi

# GPU 체크
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU 정보:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo ""
    echo "⚠️  NVIDIA GPU를 찾을 수 없습니다."
    echo "vLLM은 GPU가 필수입니다. CUDA가 설치되어 있는지 확인하세요."
    echo ""
    read -p "그래도 계속하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]
    then
        exit 1
    fi
fi

# 통합 모델 캐시(프로젝트 내부) 설정
export HF_HOME="$(pwd)/models"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

echo "================================"
echo "카피바라를 시작합니다..."
echo "잠시만 기다려주세요..."
echo "모델 캐시 디렉토리: $HF_HOME"
echo "================================"
echo ""

# Python 실행
python capybara.py

# 종료 코드 체크
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 카피바라 실행 중 오류가 발생했습니다."
    echo ""
    echo "문제 해결 방법:"
    echo "  1. GPU 메모리 부족: nvidia-smi로 확인"
    echo "  2. vLLM 버전 문제: pip install --upgrade vllm"
    echo "  3. CUDA 버전 문제: nvidia-smi로 CUDA 버전 확인"
    exit 1
fi
