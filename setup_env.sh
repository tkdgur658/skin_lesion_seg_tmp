#!/bin/bash

# ============================================
# Skin_Lesion_Seg 환경 설치 스크립트
# ============================================

echo "============================================"
echo "Creating Skin_Lesion_Seg environment..."
echo "============================================"

# Conda 환경 생성
conda env create -f environment_SLS.yml

# 환경 활성화
echo ""
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate Skin_Lesion_Seg

# Jupyter 커널 등록
echo ""
echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name Skin_Lesion_Seg --display-name "Skin_Lesion_Seg (Python 3.8)"

echo ""
echo "============================================"
echo "Setup completed!"
echo "============================================"
echo ""
echo "To activate the environment:"
echo "  conda activate Skin_Lesion_Seg"
echo ""
echo "Jupyter kernel registered as: Skin_Lesion_Seg (Python 3.8)"
echo ""
