import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture


def logfile_profiling(fpath, json_path, key):
    mask_losses = []
    prev_prompt = None    
    results = []
    avg_min_mask = []
    avg_max_mask = []
    hist_dict = {}
    name_list = []
    avg_delta_mask = []

    with open(fpath, 'r') as log_file:
        with open(json_path, 'r') as raw_json_file:
            raw_clip_dict = json.load(raw_json_file)
            for line in log_file:        
                if 'Prompt' in line:
                    prompt_name = line.split('Prompt')[1].split(': ')[1]
                    current_prompt = prompt_name.split('\n')[0]
                    if prev_prompt != current_prompt:
                        if prev_prompt is not None:
                            hist_dict[current_prompt] = {
                                'name':name_list,
                                'min':avg_min_mask,
                                'max':avg_max_mask,
                                'delta': avg_delta_mask
                            }
                            
                            avg_min_mask_loss = sum(avg_min_mask) / len(avg_min_mask)
                            avg_max_mask_loss = sum(avg_max_mask) / len(avg_max_mask)
                            avg_delta_mask_loss = sum(avg_delta_mask) / len(avg_delta_mask)
                            results.append(f'{prev_prompt} {key} minimum {avg_min_mask_loss:.4f} maximum {avg_max_mask_loss:.4f} delta {avg_delta_mask_loss:.4f} \n')
                            avg_min_mask = []
                            avg_max_mask = []
                            avg_delta_mask_loss = []
                            name_list = []


                        if current_prompt in raw_clip_dict:
                            list_full_similarities = raw_clip_dict[current_prompt]['full_text']
                            average_full_similarities = sum(list_full_similarities) / len(list_full_similarities)
                            
                            list_1st_similarities = raw_clip_dict[current_prompt]['first_half']                
                            list_2nd_similarities = raw_clip_dict[current_prompt]['second_half']
                            list_min_similarities = []

                            for i, first_half in enumerate(list_1st_similarities):
                                list_min_similarities.append(min(first_half, list_2nd_similarities[i]))
                            
                            average_min_similarities = sum(list_min_similarities) / len(list_min_similarities)
                            results.append(f'{current_prompt} avg full sim {average_full_similarities:.4f} avg min sim {average_min_similarities:.4f}\n')
                        else:
                            break

                
                if 'mask loss' in line:
                    loss_value = float(line.split('mask loss')[1].strip())
                    mask_losses.append(loss_value)

                if 'seed' in line:
                    prev_prompt = current_prompt
                    seed_number = line.split('seed')[1].split('.jpg')[0]

                    target_name = f'2n5_seed{seed_number}.jpg'

                    if target_name in raw_clip_dict[current_prompt]['image_names']:
                        index = raw_clip_dict[current_prompt]['image_names'].index(target_name)
                        full_sim = raw_clip_dict[current_prompt]['full_text'][index]
                        first_sim = raw_clip_dict[current_prompt]['first_half'][index]
                        second_sim = raw_clip_dict[current_prompt]['second_half'][index]
                        min_sim = min(first_sim, second_sim)

                    raw_clip_dict[current_prompt]
                    min_loss = min(mask_losses)
                    max_loss = max(mask_losses)
                    dif = max(mask_losses) - min(mask_losses)
                    
                    if key == 'up':
                        if full_sim > average_full_similarities and min_sim > average_min_similarities:
                            results.append(f'{current_prompt} {seed_number} min {min_loss:.4f} max {max_loss:.4f} delta {dif:.4f} full {full_sim:.4f} min {min_sim:.4f}\n')
                            avg_min_mask.append(min_loss)
                            avg_max_mask.append(max_loss)
                            avg_delta_mask.append(dif)
                            name_list.append(target_name)
                            
                    else:
                        if full_sim < average_full_similarities and min_sim < average_min_similarities:
                            results.append(f'{current_prompt} {seed_number} min {min_loss:.4f} max {max_loss:.4f} diff {dif:.4f} full {full_sim:.4f} min {min_sim:.4f}\n')
                            avg_min_mask.append(min_loss)
                            avg_max_mask.append(max_loss)
                            avg_delta_mask.append(dif)
                            name_list.append(target_name)                           

                    mask_losses = []
    return results, hist_dict

def gmm_modeling(gmm, x_start, x_end, hist_list):

    # gmm = GaussianMixture(n_components=components, random_state=random_state)
    import math
    hist_list = [x for x in hist_list if math.isnan(x) == False]
    gmm.fit(np.array(hist_list).reshape(-1, 1))

    mu = gmm.means_.flatten()
    sigma = np.sqrt(gmm.covariances_).flatten()
    alpha = gmm.weights_
    
    x = np.linspace(x_start, x_end)
    gmm_pdf = np.zeros_like(x, dtype=np.float64)
    
    for j in range(len(alpha)):
        gmm_pdf += alpha[j] * np.exp(-0.5 * ((x - mu[j]) ** 2) / (sigma[j] ** 2)) / (sigma[j] * np.sqrt(2 * np.pi))

    return gmm_pdf, mu, sigma, alpha

if __name__ == "__main__":
    import os

    fpath = '/home/initno1/GSN/mino_v002/hist_test_v003/main_log.log'
    json_path = '/home/initno1/GSN/mino_v002/hist_test_v003/20241104_090050_eval_ti-sim/clip_raw_metrics.json'
    bins = 30
    out_put_dir = '/home/initno1/GSN/mino_v002/hist_test_v003/try8'
    os.makedirs(out_put_dir, exist_ok=True)
    
    # 평균이하
    key = 'down'
    
    down_results, down_hist_dict = logfile_profiling(fpath=fpath, json_path=json_path, key=key)        

    down_min_hist = []
    down_max_hist = []
    down_delta_hist = []

    min_values = [v["min"] for v in down_hist_dict.values()]
    for min_value in min_values:
        down_min_hist.extend(min_value)
    max_values = [v["max"] for v in down_hist_dict.values()]
    for max_value in max_values:
        down_max_hist.extend(max_value)
    delta_values = [v["delta"] for v in down_hist_dict.values()]
    for delta_value in delta_values:
        down_delta_hist.extend(delta_value)    
    
    # 평균 이상
    key = 'up'
    # results_path = f'{out_put_dir}/log_results_{key}.txt'
    up_results, up_hist_dict = logfile_profiling(fpath=fpath, json_path=json_path, key=key)        

    up_min_hist = []
    up_max_hist = []
    up_delta_hist = []
    min_values = [v["min"] for v in up_hist_dict.values()]
    for min_value in min_values:
        up_min_hist.extend(min_value)
    max_values = [v["max"] for v in up_hist_dict.values()]
    for max_value in max_values:
        up_max_hist.extend(max_value)
    delta_values = [v["delta"] for v in up_hist_dict.values()]
    for delta_value in delta_values:
        up_delta_hist.extend(delta_value)

    os.environ['OMP_NUM_THREADS'] = '2'
    components = 1
    random_state = 42
    gmm = GaussianMixture(n_components=components, random_state=random_state)

    # max gmm
    # up_max_gmm_pdf, up_max_gmm_mu, up_max_gmm_sigma, up_max_gmm_alpha = gmm_modeling(gmm, x_start=0, x_end=2.0, hist_list=up_max_hist)
    # up_max_x = np.linspace(0, 2.0)
    # down_max_gmm_pdf, down_max_gmm_mu, down_max_gmm_sigma, down_max_gmm_alpha = gmm_modeling(gmm, x_start=0, x_end=2.0, hist_list=down_max_hist)
    # down_max_x = np.linspace(0, 2.0)

    # # min gmm
    # up_min_gmm_pdf, up_min_gmm_mu, up_min_gmm_sigma, up_min_gmm_alpha = gmm_modeling(gmm, x_start=0, x_end=1.5, hist_list=up_min_hist)
    # up_min_x = np.linspace(0, 1.5)
    # down_min_gmm_pdf, down_min_gmm_mu, down_min_gmm_sigma, down_min_gmm_alpha = gmm_modeling(gmm, x_start=0, x_end=1.5, hist_list=down_min_hist)
    # down_min_x = np.linspace(0, 1.5)

    # delta gmm    
    up_delta_gmm_pdf, up_delta_gmm_mu, up_delta_gmm_sigma, up_delta_gmm_alpha = gmm_modeling(gmm, x_start=0, x_end=1.5, hist_list=up_delta_hist)
    up_delta_x = np.linspace(0, 1.5)    
    down_delta_gmm_pdf, down_delta_gmm_mu, down_delta_gmm_sigma, down_delta_gmm_alpha = gmm_modeling(gmm, x_start=0, x_end=1.5, hist_list=down_delta_hist)
    down_delta_x = np.linspace(0, 1.5)    
    
    # gmm_modeling
    # gmm = GaussianMixture(n_components=components, random_state=random_state)
    # gmm.fit(np.array(up_max_hist).reshape(-1, 1))
    # mu_down_min = gmm.means_.flatten()
    # sigma_down_min = np.sqrt(gmm.covariances_).flatten()
    # alpha_down_min = gmm.weights_    
    # x = np.linspace(0, 2.0)
    # gmm_pdf = np.zeros_like(x, dtype=np.float64)
    
    # for j in range(len(alpha_down_min)):
    #     gmm_pdf += alpha_down_min[j] * np.exp(-0.5 * ((x - mu_down_min[j]) ** 2) / (sigma_down_min[j] ** 2)) / (sigma_down_min[j] * np.sqrt(2 * np.pi))
    
    # up_results.append(f'up max gmm results mu {up_max_gmm_mu} sigma {up_max_gmm_sigma} alpha {up_max_gmm_alpha}\n')
    # down_results.append(f'down max gmm results mu {down_max_gmm_mu} sigam {down_max_gmm_sigma} alpha {down_max_gmm_alpha}\n')
    
    # up_results.append(f'up min gmm results mu {up_min_gmm_mu} sigma {up_min_gmm_sigma} alpha {up_min_gmm_alpha}\n')
    # down_results.append(f'down min gmm results mu {down_min_gmm_mu} sigam {down_min_gmm_sigma} alpha {down_min_gmm_alpha}\n')

    up_results.append(f'up delta gmm results mu {up_delta_gmm_mu} sigma {up_delta_gmm_sigma} alpha {up_delta_gmm_alpha}\n')
    down_results.append(f'down delta gmm results mu {down_delta_gmm_mu} sigam {down_delta_gmm_sigma} alpha {down_delta_gmm_alpha}\n')

    
    # min histogram up vs down
    # plt.plot(down_min_x, down_min_gmm_pdf, label='down_min_gmm')
    # plt.plot(up_min_x, up_min_gmm_pdf, label='up_min_gmm')
    plt.hist(down_min_hist, alpha=0.4, label='down_min', density=False, bins=bins, histtype='bar')
    plt.hist(up_min_hist, alpha=0.4, label='up_min', density=False, bins=bins, histtype='bar')
    plt.legend()
    plt.savefig(f'{out_put_dir}/{bins}bins_minloss_up_vs_down.png')
    plt.clf()

    # down min
    # plt.plot(down_min_x, down_min_gmm_pdf)
    plt.hist(down_min_hist, label='down_min', density=False, bins=bins, histtype='bar')    
    plt.savefig(f'{out_put_dir}/{bins}bins_minloss_down.png')
    plt.clf()

    # up min
    # plt.plot(up_min_x, up_min_gmm_pdf)
    plt.hist(up_min_hist, label='up_min', bins=bins, density=False, histtype='bar')    
    plt.savefig(f'{out_put_dir}/{bins}bins_minloss_up.png')
    plt.clf()
    
    # max histogram up vs down    
    # plt.plot(down_max_x, down_max_gmm_pdf, label='down_max_gmm')
    # plt.plot(up_max_x, up_max_gmm_pdf, label='up_max_gmm')
    plt.hist(down_max_hist, alpha=0.4, label='down_min', density=False, bins=bins, histtype='bar')
    plt.hist(up_max_hist, alpha=0.4, label='up_min', density=False, bins=bins, histtype='bar')
    plt.legend()
    plt.savefig(f'{out_put_dir}/{bins}bins_maxloss_up_vs_down.png')
    plt.clf()

    # down max
    # plt.plot(down_max_x, down_max_gmm_pdf)
    plt.hist(down_max_hist, label='down_min', density=False, bins=bins, histtype='bar')    
    plt.savefig(f'{out_put_dir}/{bins}bins_max_down.png')    
    plt.clf()
    
    # up max
    # plt.plot(up_max_x, up_max_gmm_pdf)
    plt.hist(up_max_hist, label='up_min', density=False, bins=bins, histtype='bar')    
    plt.savefig(f'{out_put_dir}/{bins}bins_max_up.png')
    plt.clf()

    # delta up vs down
    plt.plot(down_delta_x, down_delta_gmm_pdf, 'down_delta_gmm')
    plt.plot(up_delta_x, up_delta_gmm_pdf, 'up_delta_gmm')
    plt.hist(down_delta_hist, alpha=0.4, label='down_delta', density=True, bins=bins, histtype='bar')
    plt.hist(up_delta_hist, alpha=0.4, label='up_delta', density=True, bins=bins, histtype='bar')
    plt.legend()
    plt.savefig(f'{out_put_dir}/{bins}bins_delta_up_vs_down.png')
    plt.clf()

    # down delta
    plt.plot(down_delta_x, down_delta_gmm_pdf)
    plt.hist(down_delta_hist, label='down_delta', density=True, bins=bins, histtype='bar')    
    plt.savefig(f'{out_put_dir}/{bins}bins_delta_down.png')
    plt.clf()    

    # up delta
    plt.plot(up_delta_x, up_delta_gmm_pdf)
    plt.hist(up_delta_hist, label='up_delta', density=True, bins=bins, histtype='bar')    
    plt.savefig(f'{out_put_dir}/{bins}bins_delta_up.png')
    plt.clf()
    
    # write txt
    results_path = f'{out_put_dir}/log_results_up.txt'
    with open(results_path, 'w') as file:
        for result in up_results:
            file.write(result)

    # write txt

    results_path = f'{out_put_dir}/log_results_down.txt'
    with open(results_path, 'w') as file:
        for result in down_results:
            file.write(result)


    