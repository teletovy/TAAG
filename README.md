# Temporal Adaptive Attention Map Guidance

Abstract: Text-to-image generation aims to create visually compelling images aligned with input prompts, but challenges such as subject mixing and subject neglect, often caused by semantic leakage during the generation process, remain, particularly in multi-subject scenarios. To mitigate this, existing methods optimize attention maps in diffusion models, using static loss functions at each time step, often leading to suboptimal results due to insufficient consideration of varying characteristics across diffusion stages. To address this problem, we propose a novel framework that adaptively guides the attention maps by dividing the diffusion process into four intervals: initial, layout, shape, and refinement. We adaptively optimize attention maps using interval-specific strategies and a dynamic loss function. Additionally, we introduce a seed filtering method based on the self-attention map analysis to detect and address the semantic leakage by restarting the generation process with new noise seeds when necessary. Extensive experiments on various datasets demonstrate that our method achieves significant improvements in generating images aligned with input prompts, outperforming} previous approaches both quantitatively and qualitatively.

 이 연구는 텍스트-이미지 생성에서 발생할 수 있는 주요 문제인 주제 혼합과 주제 무시를 해결하려는 시도입니다. 기존 방법들이 diffusion model에서 고정된 손실 함수를 사용하여 어텐션 맵을 최적화하는 한계를 넘어서기 위해, diffusion model의 denoising sampling 과정을 네 가지 단계(초기, 레이아웃, 형상, 세부화)로 나누어 각 단계에 맞는 최적화 전략과 동적 손실 함수를 적용하는 새로운 프레임워크를 제안했습니다. 또한, self-attention map을 활용한 seed filtering 방법을 통해 의미 유출 문제를 해결하고, 필요 시 새로운 noise seed를 사용하여 생성을 재시작하는 방식으로 성능을 개선했습니다. 

# Overview
![Image](https://github.com/user-attachments/assets/8c04590b-8fd4-4b00-b8b0-aaeba9093be7)

# Examples
![Image](https://github.com/user-attachments/assets/a3a8fbee-fb96-42d3-ad58-a531cb55e56d)
