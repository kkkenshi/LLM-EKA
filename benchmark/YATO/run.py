import subprocess
import datetime
dataset_name_list = ['BIORED']
lr = 3e-5
dropout = 0.3
iteration = 100
batchsize = 10
seed_list = [42,52,62]
device = 0
time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# model_name = 'COVID-TWITTER-BERT'
# plm_path = r'/gemini/pretrain2/COVID-TWITTER-BERT'

model_name = 'PubMedBERT'
plm_path = r'/gemini/pretrain3'

use_crf = False

for dataset_name in dataset_name_list:
    for seed in seed_list:
        model_dir = f"exps/{dataset_name}/m_{model_name}_s_{seed}_t_{time}/"
        log_file = f"exp_log/d_{dataset_name}_m_{model_name}_s_{seed}_t_{time}.log"

        command = [
            "python", "train_plm.py",
            "--HP_lr", str(lr),
            "--HP_dropout", str(dropout),
            "--HP_batch_size", str(batchsize),
            "--HP_iteration", str(iteration),
            "--device", str(device),
            "--dataset_name", dataset_name,
            "--high_level_transformer", plm_path,
            "--model_dir", model_dir,
            "--seed", str(seed)
            #"--use_crf" if use_crf else ""
        ]

        with open(log_file, "w") as logfile:
            subprocess.run(command, stdout=logfile, stderr=logfile)

        print(f"Training completed for dataset {dataset_name}, seed {seed}. Log saved to {log_file}")
