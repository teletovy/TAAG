import os
import torch
import time
from omegaconf import OmegaConf
from pipelines.pipeline_mino_v002 import SDMINOv002
from visu_utils.attention_visualization_utils import show_cross_attention, view_images
import cv2
from utils.code_utils import copy_src_files, get_logger, get_indices, get_time_str, get_iou_of_mask
import matplotlib.pyplot as plt

def main(cfg):
    # get config
    cfg = OmegaConf.load(cfg)

    save_path = os.path.join(cfg.result_root, get_time_str())
    copy_src_files(
        os.path.dirname(os.path.abspath(__file__)), # current dir
        os.path.join(save_path, 'src')
        )

    # load diffusers pipeline
    pipe = SDMINOv002.from_pretrained(cfg.model_id).to("cuda")    

    # get logger
    logger = get_logger(save_path)
    logger.info(f'Run {__file__} file')
    exp_start = time.time()
    
    prompt = cfg.prompt
    # save current config
    OmegaConf.save(cfg, os.path.join(save_path, 'config.yaml'))

    SEEDS = [x + cfg.start_seed for x in range(cfg.max_seed_numbers)]
    
    prompt_dir = os.path.join(save_path, prompt.replace(" ", "_"))
    os.makedirs(prompt_dir, exist_ok=True)        
    logger.info(f"Prompt: {prompt}")                
    token_indices = get_indices(prompt, logger)
    logger.info(f"token indices:{token_indices}")       

    image_cnt = 0 #NOTE 09/24 23:28 mask기반 seed selection?
        
    for seed in SEEDS:
        if image_cnt == cfg.image_nums:
            break         
    
        gen_start = time.time()
        generator = torch.Generator("cuda").manual_seed(seed)       
            
        images, results_dict, mask_anormaly = pipe(
                        prompt=prompt,
                        token_indices=token_indices,                    
                        logger=logger,
                        cfg=cfg,
                        generator=generator,
                        save_path=prompt_dir,
                        seed=seed)            
        if cfg.denoising.filtering and mask_anormaly:                
            pass

        else:
            mask_dict = results_dict['masks']
            images = images.images
            image = images[0]
            image_cnt += 1

            image.save(f"{prompt_dir}/{token_indices[0]}n{token_indices[1]}_seed{seed}.jpg")
            logger.info(f"\t saved at {prompt_dir}/{token_indices[0]}n{token_indices[1]}_seed{seed}.jpg")
            gen_end = time.time()
            logger.info(f"\t Generation one images elapsed {gen_end - gen_start:0.4f} sec")

            visu_prompt=["<|startoftext|>",] + prompt.split(' ') + ["<|endoftext|>",] 
            os.makedirs(f'{prompt_dir}/{seed}/crossattn', exist_ok=True)       

            # NOTE 10/30 deprecated
            if cfg.visu_denoising_process:
                os.makedirs(f'{prompt_dir}/{seed}/intermediates', exist_ok=True)
                for idx, inter_image in enumerate(results_dict['intermediate']):
                    inter_image[0].save(f'{prompt_dir}/{seed}/intermediates/{idx * cfg.attn_sampling_rate}th_inter.jpg')

            if cfg.visu_attn:
                for sampling_steps, cross_attn_map in enumerate(results_dict['cross']):                                
                    cross_attn_img, splited_image = show_cross_attention(prompts=visu_prompt, attention_maps=cross_attn_map)
                    cross_attn_img.save(f"{prompt_dir}/{seed}/crossattn/{sampling_steps * cfg.attn_sampling_rate}th_attn.png")
                    for subject_index in token_indices:
                        splited_pil_img = view_images(splited_image[subject_index], display_image=False)
                        splited_pil_img.save(f"{prompt_dir}/{seed}/crossattn/{sampling_steps * cfg.attn_sampling_rate}th_idx{subject_index}_attn.png")

            #NOTE 1127 paper figure
            if cfg.visu_mask_construct:
                os.makedirs(f'{prompt_dir}/{seed}/mask_construct', exist_ok=True)       
                mask_cross_attn_maps = results_dict['mask_construct']['cross']
                mask_cross_attn_img, mask_cross_splited_image = show_cross_attention(prompts=visu_prompt, attention_maps=mask_cross_attn_maps)
                for subject_index in token_indices:
                    splited_pil_img = view_images(mask_cross_splited_image[subject_index], display_image=False)
                    splited_pil_img.save(f"{prompt_dir}/{seed}/mask_construct/idx{subject_index}_mask_construct_attn.png")
                
                cluster_maps = results_dict['mask_construct']['cluster']
                plt.imshow(cluster_maps)
                plt.axis('off')
                plt.savefig(f"{prompt_dir}/{seed}/mask_construct/cluster_maps.png", bbox_inches='tight', pad_inches=0)

            for i in range(len(token_indices)):
                for j in range(i+1, len(token_indices)):
                    mask_iou = get_iou_of_mask(mask_dict[token_indices[i] - 1], mask_dict[token_indices[j] - 1])
            
            for k, v in results_dict['masks'].items():
                # evaluation mask                
                cv2.imwrite(f'{prompt_dir}/{seed}/{k}_{mask_iou:0.3f}_mask.jpg', v * 255)

    exp_end = time.time()
    elapse = exp_end - exp_start
    hours = elapse // 3600
    min = (elapse % 3600) // 60
    sec = (elapse % 3600) % 60
    logger.info(f'\t exp finished {hours}h {min}min {sec}sec')


if __name__ == '__main__':
    cfg_path = '/home/initno1/GSN/mino_v002/conf/conf_mino_v002_single.yaml'
    main(cfg_path)


#NOTE 1109 
'mino v002와 동일한 버전인데 attribute binindg을 위한 버전'