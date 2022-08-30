import os
from pathlib import Path
import jellyfish

# this file is for checking most similar names of masks to missing masks

masks_path = Path('dataset/masks')

masks = [mask_name for mask_name in os.listdir(masks_path) if mask_name.endswith('.vpa')]

missing_file = '96-6433-1-L-20X-0.50 ph1__UPlanF1__Olympus.vpa'

closest = sorted(masks, key=lambda x: jellyfish.levenshtein_distance(x, missing_file))
for name in closest:
    print(name)
