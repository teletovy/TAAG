import json

# NOTE 11/15 area-> initno base-> mine
name = 'fin'
base_clip_json_path = '/home/initno1/exp/mino_nsfw/mino/20241109_101911_nsfw_eval_ti-sim/clip_raw_metrics.json'
base_blip_json_path = '/home/initno1/exp/mino_nsfw/mino/20241109_101911_nsfw_eval_blip_tt_sim/blip_raw_metrics.json'

area_clip_json_path = '/home/initno1/exp/mino_v002_ablation/20241201_133203_eval_ti-sim/clip_raw_metrics.json'
area_blip_json_path = '/home/initno1/exp/mino_v002_ablation/20241201_133203_eval_blip_tt_sim/blip_raw_metrics.json'

with open(base_blip_json_path, 'r') as file:
    base_blip_data = json.load(file)

with open(base_clip_json_path, 'r') as file:
    base_clip_data = json.load(file)

with open(area_blip_json_path, 'r') as file:
    area_blip_data = json.load(file)

with open(area_clip_json_path, 'r') as file:
    area_clip_data = json.load(file)

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
        if img_file in area_clip_data[prompt]['image_names']:
            area_clip_index = area_clip_data[prompt]['image_names'].index(img_file)
            # area_clip_index = area_clip_data[prompt]['image_names'].index(target_initno_name)
            area_full_sim = area_clip_data[prompt]['full_text'][area_clip_index]
            area_1st_sim = area_clip_data[prompt]['first_half'][area_clip_index]
            area_2nd_sim = area_clip_data[prompt]['second_half'][area_clip_index]
            area_min_sim = min(area_1st_sim, area_2nd_sim)
            #initno blip score
            if img_file in area_blip_data[prompt]['image_names']:
            # if target_initno_name in area_blip_data[prompt]['image_names']:
                area_blip_index = area_blip_data[prompt]['image_names'].index(img_file)
                # area_blip_index = area_blip_data[prompt]['image_names'].index(target_initno_name)
                area_blip_sim = area_blip_data[prompt]['text_similarities'][area_blip_index]     

                # comparison
                # comparison_full = base_full_sim < area_full_sim
                # comparison_min = base_min_sim < area_min_sim
                comparison_blip = base_blip_sim > area_blip_sim
                
                # if comparison_min and comparison_blip:
                if comparison_blip:
                    results.append(f'prompt {prompt} name {img_file} base F{base_full_sim:.4f} M{base_min_sim:.4f} B{base_blip_sim:.4f} area F{area_full_sim:.4f} M{area_min_sim:.4f} B{area_blip_sim:.4f}\n')

            else:
                ValueError(f'invalid name {img_file}')        
        else:
            print('because of filtering')
            pass
        
results_path = '/home/initno1/exp/quan_analysis/ablation_comparison_3/'
import os
os.makedirs(results_path, exist_ok=True)
with open(os.path.join(results_path,f'm1_vs_m2_{name}.txt'), 'w') as file:
    for result in results:
        file.write(result)