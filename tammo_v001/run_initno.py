import os
import torch
import time
from omegaconf import OmegaConf


from pipelines.pipeline_initno import StableDiffusionInitNOPipeline
from visu_utils.attention_visualization_utils import show_cross_attention
import cv2
from utils.code_utils import copy_src_files, get_logger, get_indices, get_time_str, get_iou_of_mask
# from visu_utils.attention_visualization_utils import show_cross_attention, show_self_attention

def main(cfg):
    # get config
    cfg = OmegaConf.load(cfg)
    # get exp name
    save_path = os.path.join(cfg.result_root, get_time_str())
    copy_src_files(
        os.path.dirname(os.path.abspath(__file__)), # current dir
        os.path.join(save_path, 'src')
        )

    # load diffusers pipeline
    pipe = StableDiffusionInitNOPipeline.from_pretrained(cfg.model_id).to("cuda")    

    # get logger
    logger = get_logger(save_path)
    logger.info(f'Run {__file__} file')
    exp_start = time.time()

    # prompt & seed number
    if cfg.benchmark == 'ane':
        from utils.ane_prompt import AnEPrompt
        PROMPTS = AnEPrompt()
    elif cfg.benchmark == 'dnb':
        from utils.dnb_prompt import DnBPrompt
        PROMPTS = DnBPrompt()
    else:
        ValueError(f'Invalid key {cfg.benchmark}')

    SEEDS = [x for x in range(cfg.init_seed_num, cfg.init_seed_num + cfg.max_seed_numbers)]

    # save current config
    OmegaConf.save(cfg, os.path.join(save_path, 'config.yaml'))   

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
            token_indices = get_indices(prompt, logger)
        elif cfg.subset_key == 'animals_scene':
            token_indices = [2, 5]
        elif cfg.subset_key == 'color_objects_scene':
            token_indices = [3, 7]
        elif cfg.subset_key == 'multi_objects':
            token_indices = get_indices(prompt, logger)

        logger.info(f"token indices:{token_indices}")
        image_cnt = 0
        avg_time_per_seed = []

        for SEED in SEEDS:
            if image_cnt == cfg.image_nums:
                assert len(avg_time_per_seed) == cfg.image_nums
                logger.info(f'generate {cfg.image_nums} image complete, check {len(avg_time_per_seed)}')
                logger.info(f'prompt {prompt} avg time {sum(avg_time_per_seed)/len(avg_time_per_seed)}')
                avg_time_per_prompt.append(sum(avg_time_per_seed)/len(avg_time_per_seed))
                break            
            
            logger.info('Seed ({}) Processing the ({}) prompt'.format(SEED, prompt))            
            
            generator = torch.Generator("cuda").manual_seed(SEED)
            gen_start = time.time()            
            outputs, results_dict = pipe(
                prompt=prompt,
                token_indices=token_indices,                
                generator=generator,
                # result_root=result_root,
                seed=SEED,                
                logger=logger,
                cfg=cfg
            )   
            gen_end = time.time()
            if outputs.nsfw_content_detected[0]:
                logger.info(f'nsfw contents has detected {outputs.nsfw_content_detected}')
                pass
            else:                
                images = outputs.images
                image = images[0]
                image_cnt += 1
                naming = "n".join(map(str, token_indices))
                image.save(f"{prompt_dir}/InitNO_{naming}_seed{SEED}.jpg")
                logger.info(f"\t saved at {prompt_dir}/InitNO_{token_indices[0]}n{token_indices[1]}_seed{SEED}.jpg")
                
                avg_time_per_seed.append(gen_end - gen_start)
                logger.info(f"\t Generation a image elapsed {gen_end - gen_start:0.4f} sec")

                visu_prompt=["<|startoftext|>",] + prompt.split(' ') + ["<|endoftext|>",] 
                os.makedirs(f'{prompt_dir}/{SEED}/crossattn', exist_ok=True)            

                if cfg.visu_attn:
                    for sampling_steps, cross_attn_map in enumerate(results_dict['cross']):                                
                        cross_attn_img, _ = show_cross_attention(prompts=visu_prompt, attention_maps=cross_attn_map)
                        cross_attn_img.save(f"{prompt_dir}/{SEED}/crossattn/{sampling_steps}th_attn.png")                
                
                for i in range(len(token_indices)):
                        for j in range(i+1, len(token_indices)):
                            mask_iou = get_iou_of_mask(results_dict['masks'][token_indices[i] - 1], results_dict['masks'][token_indices[j] - 1])                
                
                for k, v in results_dict['masks'].items():
                    # evaluation mask                
                    cv2.imwrite(f'{prompt_dir}/{SEED}/{k}_mask.jpg', v * 255)

    exp_end = time.time()
    logger.info(f'\t exp finished {exp_end - exp_start:0.4f} sec')
    
    

if __name__ == '__main__':
    cfg_path = '/home/initno1/GSN/mino_v002/conf/conf_initno.yaml'
    main(cfg_path)