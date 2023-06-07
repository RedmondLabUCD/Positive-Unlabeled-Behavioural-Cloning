# Positive Unlabeled Behavioural Cloning (PUBC)
This code has been developed to reproduce the PUBC methodology described in our paper titled **Improving Behavioural Cloning with Positive Unlabeled Learning**. We have included two of the most challenging real-world physical robotic manipulation tasks in this repository for demonstration.

# Installation
    pip install -r requirements.txt
    
# Reproduce PUBC
Firstly, you need to visit https://drive.google.com/drive/folders/16q16012lGAlsnChrKUL6nR5eTODax5aq?usp=sharing to download the datasets to your local device.

If you want only run the PU learning(filter part) without policy learning:

    python main.py  --raw-dataset-path='<path to the mixed dataset>'  --pos-seed-dataset-path='<path to the seed dataset>'  --train-policy=False
    
If you want only run full PUBC:

    python main.py  --raw-dataset-path='<path to the mixed dataset>'  --pos-seed-dataset-path='<path to the seed dataset>'  --train-policy=True --policy='bc'

# Evaluate PUBC
To submit the trained policy model for evaluation on a real robot cluster, please follow the instructions provided on this page: https://webdav.tuebingen.mpg.de/trifinger-rl/docs/real_robot/submission_system.html
