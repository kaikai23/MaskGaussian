import os

data_dir = './data/gs_datasets'
save_dir = './output'

subdirectories = [
    "bicycle",  
    "bonsai",
    "counter",  
    "drjohnson",
    "flowers",
    "garden",
    "kitchen",  
    "playroom",  
    "room",
    "stump",  
    "train", 
    "treehill",  
    "truck",
]
data_device = 'cuda'

# Ours-α
# lambda_mask = 0.1
# mask_from_iter = 19000
# mask_until_iter = 20000

# Ours-β
lambda_mask = 0.0005
mask_from_iter = 0
mask_until_iter = 30000

# Ours-γ
# lambda_mask = 0.001
# mask_from_iter = 0
# mask_until_iter = 30000


for d in subdirectories:
    input_dir = os.path.join(data_dir, d)
    output_dir = os.path.join(save_dir, d + f'_{lambda_mask}_{mask_from_iter}_{mask_until_iter}_{data_device}')
    os.makedirs(output_dir, exist_ok=True)

    # train
    command = f"python train.py \
    -s {input_dir} \
    -m {output_dir} \
    --eval \
    --lambda_mask {lambda_mask} \
    --mask_from_iter {mask_from_iter} \
    --mask_until_iter {mask_until_iter} \
    --data_device {data_device} >> {output_dir}/train.log 2>&1"
    print(f"Running: {command}")
    os.system(command)

    # render test images
    command = f"python render.py -m {output_dir} --skip_train"
    print(f"Running: {command}")
    os.system(command)

    # evaluate test images
    command  = f"python metrics.py -m {output_dir}"
    print(f"Running: {command}")
    os.system(command)