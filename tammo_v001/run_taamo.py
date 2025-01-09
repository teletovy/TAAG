#NOTE 1223 official version을 위해 naming 변경, 다양한 benchmark 수용하도록 code 변경

import os
import torch
import time
from omegaconf import OmegaConf
from pipelines.pipeline_tammo import TAMMOPipeline
from visu_utils.attention_visualization_utils import show_cross_attention
import cv2
from utils.code_utils import copy_src_files, get_logger, get_indices, get_time_str, get_iou_of_mask

def main(cfg):    
    cfg = OmegaConf.load(cfg) # get config

    save_path = os.path.join(cfg.result_root, get_time_str())
    copy_src_files(
        os.path.dirname(os.path.abspath(__file__)), # current dir
        os.path.join(save_path, 'src')
        )
    
    pipe = TAMMOPipeline.from_pretrained(cfg.model_id).to("cuda")  # load diffusers pipeline
    
    logger = get_logger(save_path) # get logger
    logger.info(f'Run {__file__} file')
    exp_start = time.time()
    
    if cfg.benchmark == 'ane':
        from utils.ane_dataset import AnEPrompt
        PROMPTS = AnEPrompt()

    elif cfg.benchmark == 'dnb':
        from utils.dnb_prompt import DnBPrompt
        PROMPTS = DnBPrompt()
    
    # save current config
    OmegaConf.save(cfg, os.path.join(save_path, 'config.yaml'))
    SEEDS = [x for x in range(cfg.max_seed_numbers)]
    avg_time_per_prompt = []

    for prompt_idx, prompt in enumerate(PROMPTS(cfg.subset_key)):    
        # NOTE 나중에 지우거나 주석처리, time 측정용
        if prompt_idx > 9:
            assert len(avg_time_per_prompt) == 10
            logger.info(f'10 prompts are completed.')                        
            logger.info(f'Total avg time {sum(avg_time_per_prompt)/len(avg_time_per_prompt)}')
            break

        prompt_dir = os.path.join(save_path, prompt.replace(" ", "_"))
        os.makedirs(prompt_dir, exist_ok=True)        
        logger.info(f"Prompt: {prompt}")        

        if cfg.benchmark == 'ane':
            token_indices, num_of_subjects = get_indices(prompt, logger)
        elif cfg.subset_key == 'animals_scene':
            token_indices = [2, 5]
            num_of_subjects = [1, 1]
        elif cfg.subset_key == 'color_objects_scene':
            token_indices = [3, 7]
            num_of_subjects = [1, 1]
        elif cfg.subset_key == 'multi_objects':
            token_indices, num_of_subjects = get_indices(prompt, logger)

        logger.info(f"token indices:{token_indices}")          
        image_cnt = 0 #NOTE 09/24 23:28 mask기반 seed selection?
        avg_time_per_seed = []
        for SEED in SEEDS:            
            if image_cnt == cfg.image_nums:
                assert len(avg_time_per_seed) == cfg.image_nums
                logger.info(f'generate {cfg.image_nums} image complete, check {len(avg_time_per_seed)}')
                logger.info(f'prompt {prompt} avg time {sum(avg_time_per_seed)/len(avg_time_per_seed)}')
                avg_time_per_prompt.append(sum(avg_time_per_seed)/len(avg_time_per_seed))
                break            
            
            generator = torch.Generator("cuda").manual_seed(SEED)       
            gen_start = time.time()            
            outputs, results_dict, mask_anormaly = pipe(
                        prompt=prompt,
                        token_indices=token_indices,                                        
                        logger=logger,
                        cfg=cfg,
                        generator=generator,
                        save_path=prompt_dir,
                        seed=SEED,
                        num_of_subjects=num_of_subjects
                        # start_seed_num=start_seed_number
                    )            
            
            if (cfg.denoising.filtering and mask_anormaly) or outputs.nsfw_content_detected[0]:                
                logger.info(f'nsfw {outputs.nsfw_content_detected[0]}')
                logger.info(f'seed filtering {mask_anormaly}')  
                pass

            else:
                gen_end = time.time() #NOTE filtering 시간도 고려
                avg_time_per_seed.append(gen_end - gen_start)
                mask_dict = results_dict['masks']
                image = outputs.images[0]                
                image_cnt += 1
                naming = "n".join(map(str, token_indices))
                image.save(f"{prompt_dir}/TAAMO_{naming}_seed{SEED}.jpg")
                logger.info(f"\t saved at {prompt_dir}/TAAMO_{naming}_seed{SEED}.jpg")
                
                logger.info(f"\t Generation one images elapsed {gen_end - gen_start:0.4f} sec")

                if cfg.visu_denoising_process:
                    os.makedirs(f'{prompt_dir}/{SEED}/intermediates', exist_ok=True)
                    for idx, inter_image in enumerate(results_dict['intermediate']):
                        inter_image[0].save(f'{prompt_dir}/{SEED}/intermediates/{idx * cfg.attn_sampling_rate}th_inter.jpg')
                
                if cfg.visu_attn:
                    visu_prompt=["<|startoftext|>",] + prompt.split(' ') + ["<|endoftext|>",] 
                    os.makedirs(f'{prompt_dir}/{SEED}/crossattn', exist_ok=True)
                    for sampling_steps, cross_attn_map in enumerate(results_dict['cross']):                                
                        cross_attn_img, _ = show_cross_attention(prompts=visu_prompt, attention_maps=cross_attn_map)
                        cross_attn_img.save(f"{prompt_dir}/{SEED}/crossattn/{sampling_steps}th_attn.png")           

                if cfg.visu_mask_construct:
                    for i in range(len(token_indices)):
                        for j in range(i+1, len(token_indices)):
                            mask_iou = get_iou_of_mask(mask_dict[token_indices[i] - 1], mask_dict[token_indices[j] - 1])
                    
                    for k, v in results_dict['masks'].items():
                        # evaluation mask                
                        cv2.imwrite(f'{prompt_dir}/{SEED}/{k}_{mask_iou:0.3f}_mask.jpg', v * 255)                

    exp_end = time.time()
    elapse = exp_end - exp_start
    hours = elapse // 3600
    min = (elapse % 3600) // 60
    sec = (elapse % 3600) % 60
    logger.info(f'\t exp finished {hours}h {min}min {sec}sec')


if __name__ == '__main__':
    cfg_path = '/home/initno1/GSN/mino_v002/conf/conf_tammo.yaml'
    main(cfg_path)
