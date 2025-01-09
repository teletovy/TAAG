import json

# NOTE 11/15 area-> initno base-> mine
name = 'fin'
base_clip_json_path = '/home/initno1/exp/mino_nsfw/mino/20241109_101911_nsfw_eval_ti-sim/clip_raw_metrics.json'
base_blip_json_path = '/home/initno1/exp/mino_nsfw/mino/20241109_101911_nsfw_eval_blip_tt_sim/blip_raw_metrics.json'

m3_clip_json_path = '/home/initno1/exp/mino_v002_ablation/20241201_133203_eval_ti-sim/clip_raw_metrics.json'
m3_blip_json_path = '/home/initno1/exp/mino_v002_ablation/20241201_133203_eval_blip_tt_sim/blip_raw_metrics.json'

m4_clip_json_path = '/home/initno1/exp/quan_analysis/ablation_comparison_4/20241130_031655_eval_ti-sim/clip_raw_metrics.json'
m4_blip_json_path = '/home/initno1/exp/quan_analysis/ablation_comparison_4/20241130_031655_eval_blip_tt_sim/blip_raw_metrics.json'

with open(base_blip_json_path, 'r') as file:
    base_blip_data = json.load(file)

with open(base_clip_json_path, 'r') as file:
    base_clip_data = json.load(file)

with open(m3_blip_json_path, 'r') as file:
    m3_blip_data = json.load(file)

with open(m3_clip_json_path, 'r') as file:
    m3_clip_data = json.load(file)

with open(m4_blip_json_path, 'r') as file:
    m4_blip_data = json.load(file)

with open(m4_clip_json_path, 'r') as file:
    m4_clip_data = json.load(file)

results = []

for prompt, dict_per_prompt in base_clip_data.items():
    for ours_idx, img_file in enumerate(dict_per_prompt['image_names']):
        seed_number = img_file.split('seed')[-1].split('.')[0]        
        target_initno_name = f'initno_{prompt}_seed{seed_number}.jpg'

        #ours clip score
        base_full_sim = dict_per_prompt['full_text'][ours_idx]
        base_first_sim = dict_per_prompt['first_half'][ours_idx]
        base_2nd_sim = dict_per_prompt['second_half'][ours_idx]
        base_min_sim = min(base_first_sim, base_2nd_sim)

        #ours blip score
        if img_file in base_blip_data[prompt]['image_names']:
            base_blip_index = base_blip_data[prompt]['image_names'].index(img_file)
            base_blip_sim = base_blip_data[prompt]['text_similarities'][base_blip_index]
        else:
            ValueError(f'invalid name {img_file}')        

        #initno clip score        
        # if target_initno_name in area_clip_data[prompt]['image_names']:
        if img_file in m3_clip_data[prompt]['image_names'] and img_file in m4_clip_data[prompt]['image_names']:

            m3_clip_index = m3_clip_data[prompt]['image_names'].index(img_file)            
            m3_full_sim = m3_clip_data[prompt]['full_text'][m3_clip_index]
            m3_1st_sim = m3_clip_data[prompt]['first_half'][m3_clip_index]
            m3_2nd_sim = m3_clip_data[prompt]['second_half'][m3_clip_index]
            m3_min_sim = min(m3_1st_sim, m3_2nd_sim)
            
            m4_clip_index = m4_clip_data[prompt]['image_names'].index(img_file)            
            m4_full_sim = m4_clip_data[prompt]['full_text'][m4_clip_index]
            m4_1st_sim = m4_clip_data[prompt]['first_half'][m4_clip_index]
            m4_2nd_sim = m4_clip_data[prompt]['second_half'][m4_clip_index]
            m4_min_sim = min(m4_1st_sim, m4_2nd_sim)
                        
            if img_file in m3_blip_data[prompt]['image_names'] and img_file in m4_blip_data[prompt]['image_names']:
            
                m3_blip_index = m3_blip_data[prompt]['image_names'].index(img_file)
                m4_blip_index = m4_blip_data[prompt]['image_names'].index(img_file)
                
                m3_blip_sim = m3_blip_data[prompt]['text_similarities'][m3_blip_index]     
                m4_blip_sim = m4_blip_data[prompt]['text_similarities'][m4_blip_index]     

                
                # comparison_blip = (base_blip_sim > m3_blip_sim) and 
                save = (base_full_sim > m4_full_sim and base_min_sim > m4_min_sim and base_blip_sim > m4_blip_sim)                
                
                if save:
                    # results.append(f'prompt {prompt} name {img_file} M2 {base_full_sim:.4f}/{base_min_sim:.4f}/{base_blip_sim:.4f} M3 {m3_full_sim:.4f}/{m3_min_sim:.4f}/{m3_blip_sim:.4f} M4 {m4_full_sim:.4f}/{m4_min_sim:.4f}/{m4_blip_sim:.4f}\n')
                    results.append(f'prompt {prompt} name {img_file} M2 {base_full_sim:.4f}/{base_min_sim:.4f}/{base_blip_sim:.4f} M4 {m4_full_sim:.4f}/{m4_min_sim:.4f}/{m4_blip_sim:.4f}\n')

            else:
                ValueError(f'invalid name {img_file}')        
        else:
            print('because of filtering')
            pass
        
results_path = '/home/initno1/exp/quan_analysis/ablation_comparison_4/'
import os
os.makedirs(results_path, exist_ok=True)
with open(os.path.join(results_path,f'm2_vs_m3_vs_m4.txt'), 'w') as file:
    for result in results:
        file.write(result)