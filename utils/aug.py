import numpy as np


def cutout(img, max_cuts=3, max_length_multiplier=5):
    # max_cuts: alpha for Cutmix
    # max_length_multiplier = beta for Cutmix
    """
    # Function: RandomCrop (ZeroPadded (4, 4)) + random occulusion image
    # Arguments:
        img: image
    # Returns:
        img
    """
    # img = bgr(img)
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    MAX_CUTS = max_cuts  # chance to get more cuts
    MAX_LENGTH_MUTIPLIER = max_length_multiplier  # chance to get larger cuts
    # 16 for cifar10, 8 for cifar100

    img *= 1. / 255

    mask = np.ones((height, width, channels), dtype=np.float32)
    nb_cuts = np.random.randint(0, MAX_CUTS + 1)

    # cutout
    for i in range(nb_cuts):
        y = np.random.randint(height)
        x = np.random.randint(width)
        length = 4 * np.random.randint(1, MAX_LENGTH_MUTIPLIER + 1)

        y1 = np.clip(y - length // 2, 0, height)
        y2 = np.clip(y + length // 2, 0, height)
        x1 = np.clip(x - length // 2, 0, width)
        x2 = np.clip(x + length // 2, 0, width)

        mask[y1:y2, x1:x2, :] = 0.

    img = img * mask
    return img
