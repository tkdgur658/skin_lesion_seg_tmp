{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dd800718",
   "metadata": {},
   "outputs": [],
   "source": [
    "# LightweightChannelUNet 파라미터 개수 확인\n",
    "\n",
    "# 모듈 임포트\n",
    "from DepthwiseChannel import LightweightChannelUNet, count_parameters\n",
    "import torch\n",
    "\n",
    "# 모델 설정\n",
    "model_config = {\n",
    "    'num_classes': 1, \n",
    "    'input_channels': 3, \n",
    "    'c_list': [8,16,24,32,48,64], \n",
    "    'bridge': True,\n",
    "    'gt_ds': True,\n",
    "}\n",
    "\n",
    "# 모델 생성\n",
    "print(\"=== LightweightChannelUNet Parameter Analysis ===\")\n",
    "model = LightweightChannelUNet(**model_config)\n",
    "\n",
    "# 파라미터 개수 확인\n",
    "total_params, trainable_params = count_parameters(model)\n",
    "\n",
    "# 추가 정보 출력\n",
    "print(f\"\\nModel Configuration:\")\n",
    "print(f\"- Input channels: {model_config['input_channels']} (RGB)\")\n",
    "print(f\"- Output classes: {model_config['num_classes']}\")\n",
    "print(f\"- Channel list: {model_config['c_list']}\")\n",
    "print(f\"- Bridge connections: {model_config['bridge']}\")\n",
    "print(f\"- Deep supervision: {model_config['gt_ds']}\")\n",
    "\n",
    "# 테스트 입력으로 모델 동작 확인\n",
    "print(f\"\\n=== Model Test ===\")\n",
    "test_input = torch.randn(1, 3, 256, 256)\n",
    "print(f\"Test input shape: {test_input.shape}\")\n",
    "\n",
    "# 평가 모드 테스트\n",
    "model.eval()\n",
    "with torch.no_grad():\n",
    "    output = model(test_input)\n",
    "    print(f\"Evaluation mode output shape: {output.shape}\")\n",
    "\n",
    "# 학습 모드 테스트 (Deep Supervision)\n",
    "model.train()\n",
    "with torch.no_grad():\n",
    "    ds_outputs, final_output = model(test_input)\n",
    "    print(f\"Training mode - Final output shape: {final_output.shape}\")\n",
    "    print(f\"Training mode - Deep supervision outputs: {len(ds_outputs)}\")\n",
    "    for i, ds_out in enumerate(ds_outputs):\n",
    "        print(f\"  DS output {i+1} shape: {ds_out.shape}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "75155b5c",
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
