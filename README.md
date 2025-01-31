# Temporal Adaptive Attention Map Guidance
![image](https://github.com/user-attachments/assets/fe7c6819-433c-4cbb-9f6e-3c965ece9c34)

Abstract: Text-to-image generation aims to create visually compelling images aligned with input prompts, but challenges such as subject mixing and subject neglect, often caused by semantic leakage during the generation process, remain, particularly in multi-subject scenarios. To mitigate this, existing methods optimize attention maps in diffusion models, using static loss functions at each time step, often leading to suboptimal results due to insufficient consideration of varying characteristics across diffusion stages. To address this problem, we propose a novel framework that adaptively guides the attention maps by dividing the diffusion process into four intervals: initial, layout, shape, and refinement. We adaptively optimize attention maps using interval-specific strategies and a dynamic loss function. Additionally, we introduce a seed filtering method based on the self-attention map analysis to detect and address the semantic leakage by restarting the generation process with new noise seeds when necessary. Extensive experiments on various datasets demonstrate that our method achieves significant improvements in generating images aligned with input prompts, outperforming} previous approaches both quantitatively and qualitatively.

# Requirements
# Executions
