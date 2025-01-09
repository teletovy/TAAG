if __name__ == '__main__':
    import cv2
    import numpy as np
    # path = '/home/initno1/exp/mask_v002/20240924_084924/a_dog_and_a_horse/8/4_0.000_mask.jpg' # problematic mask
    path = '/home/initno1/exp/mask_v002/20240924_084924/a_dog_and_a_horse/8/1_0.000_mask.jpg' # none problem
    mask_np = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask_np = np.where(mask_np > 127.5, 255, 0)
    mask_np = mask_np / 255
    mask_np = mask_np.astype(np.uint8)
    
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np)

    dst = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)

    for i in range(1, cnt):
        x, y, w, h, area = stats[i]
        cv2.rectangle(dst, (x, y), (x+w, y+h), (0, 255, 255))
    cv2.imwrite('./mask_normal_cc.jpg', dst)
    print('end')

    # log