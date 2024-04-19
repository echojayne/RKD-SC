# A Semantic Communication System Based on Robust Knowledge Distillation

This paper proposes a semantic communication system based on robust knowledge distillation (RKD-SC) for the lightweight application of large-scale semantic coding networks. We identify the problem of degraded robustness to noise in deep joint source and channel coding (JSCC) systems that are not trained jointly and propose a channel-aware autoencoder (CAA) based on the Transformer architecture to enhance the robustness of knowledge distillation-based semantic communication systems to channel noise.

## Requirements

- certifi
- ftfy
- numpy
- Pillow
- regex
- torch
- torchvision
- typing-extensions
- wcwidth

You can install the required packages using the following command:
```bash
pip install -r requirements.txt
```

## Getting Started

The code for all experiments is in the experiments folder, you can run the jupyter notebook to try the code.

- distill_student.ipynb: distill the teacher model to student,
- e2e_train_student.ipynb: train the ViT-Small model and semantic decoder end-to-end,
- joint_train_with_ae.ipynb: train the CAA and decoder jointly,
- JSCC.ipynb: the experiments of ResNet18, 
- retrain_teacher.ipynb: fine-tune the teacher,
- test_on_SNR.ipynb: test the model on multiple SNR levels.

## Result

![图片1](./results/curves/acc_on_SNR_KD_Joint.png) | ![图片2](./results/curves/acc_on_SNR_KD_Rayleigh.png) | ![图片3](./results/curves/acc_on_SNR_Joint.png)
--- | ---
![图片4](./results/curves/acc_on_SNR_ResNet.png) | ![图片5](./results/curves/acc_on_SNR_Teacher_Student.png) | ![图片6](./results/curves/Diff_SNR_Queue.png) 
