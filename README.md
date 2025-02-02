# Temporal Adaptive Attention Map Guidance

Abstract: Text-to-image generation aims to create visually compelling images aligned with input prompts, but challenges such as subject mixing and subject neglect, often caused by semantic leakage during the generation process, remain, particularly in multi-subject scenarios. To mitigate this, existing methods optimize attention maps in diffusion models, using static loss functions at each time step, often leading to suboptimal results due to insufficient consideration of varying characteristics across diffusion stages. To address this problem, we propose a novel framework that adaptively guides the attention maps by dividing the diffusion process into four intervals: initial, layout, shape, and refinement. We adaptively optimize attention maps using interval-specific strategies and a dynamic loss function. Additionally, we introduce a seed filtering method based on the self-attention map analysis to detect and address the semantic leakage by restarting the generation process with new noise seeds when necessary. Extensive experiments on various datasets demonstrate that our method achieves significant improvements in generating images aligned with input prompts, outperforming} previous approaches both quantitatively and qualitatively.
(https://www.mdpi.com/2079-9292/14/3/412)
# Problem statement
Existing works still suffers from semantic leakage, which includes subject neglect, subject mixing, and wrong attribute binding
![Image](https://github.com/user-attachments/assets/9cbeedf8-8442-4f33-936d-196326a9240d)
# Prpopsed method
Leveraging temporal property of text embedding in diffusion denoising sampling process, we adaptively optimize the latent by dynamic loss.
![Image](https://github.com/user-attachments/assets/8c04590b-8fd4-4b00-b8b0-aaeba9093be7)

# Results
![Image](https://github.com/user-attachments/assets/a3a8fbee-fb96-42d3-ad58-a531cb55e56d)
