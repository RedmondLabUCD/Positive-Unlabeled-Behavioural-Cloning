# Positive Unlabeled Behavioural Cloning (PUBC)
This code has been developed to reproduce the PUBC methodology described in our paper titled **Improving Behavioural Cloning with Positive Unlabeled Learning**. We have included two of the most challenging real-world physical robotic manipulation tasks in this repository for demonstration.

# Installation
    pip install -r requirements.txt
    
# Reproduce PUBC
Firstly, you need to visit [THIS LINK](https://zenodo.org/records/13228248?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjY0MTZjNGM3LWRkNDEtNGNkNC1hNGYzLTY1MTNlNjg4ZjcxNSIsImRhdGEiOnt9LCJyYW5kb20iOiJjMzEwMDI3ODU0N2Y1ODJhMzllYmM0Y2Q1M2FhMTNiYyJ9.e6Z_A1w5dZgUS-UB3h53IvL1CpYYuH5E4sPTmy8ko47WdtIPM2zRfTp7NhKn4l6JLAhoRslbOfwOeXrBEvQZAg) to download the datasets to your local device.

If you want only run the PU learning(filter part) without policy learning:

    python main.py  --raw-dataset-path='<path to the mixed dataset>'  --pos-seed-dataset-path='<path to the seed dataset>'  --train-policy=False
    
If you want only run full PUBC:

    python main.py  --raw-dataset-path='<path to the mixed dataset>'  --pos-seed-dataset-path='<path to the seed dataset>'  --train-policy=True --policy='bc'

If you want to use the trained filter for traing BC:

    python main.py  --raw-dataset-path='<path to the mixed dataset>'  --pos-seed-dataset-path='<path to the seed dataset>'  --load-trained-filter=True  --trained-filter-path=='<path to the trained models>'  --ckpt-iterations=<iterations number for training the trained models>  --train-policy=True --policy='bc' 


# Evaluate PUBC
To submit the trained policy model for evaluation on a real robot cluster, please follow the instructions provided on this page: https://webdav.tuebingen.mpg.de/trifinger-rl/docs/real_robot/submission_system.html

