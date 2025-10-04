dataset_name_list=('clinical')
lr=3e-5
dropout=0.3
hidden_size=1024
iteration=100
batchsize=32
seed_list=(22 32 42 52 62)
device=0
time=$(date "+%Y%m%d-%H%M%S")
#seed_list=(42)
model_name='bert-large-cased'
plm_path='../../pretrained_models/bert/bert-large-cased'
for dataset_name in ${dataset_name_list[@]}
do
  for seed in ${seed_list[@]}
  do
    python train_plm.py --HP_hidden_dim=${hidden_size} --HP_lr=${lr} --HP_dropout=${dropout} --HP_batch_size=${batchsize} --HP_iteration=${iteration} --device=${device} --dataset_name=${dataset_name} --high_level_transformer=${plm_path} --model_dir=exps/${dataset_name}/m_${model_name}_s_${seed}_t_${time}/ --seed=${seed} > exp_log/d_${dataset_name}_m_${model_name}_s_${seed}_t_${time}.log
  done
done
#
model_name='bert-large-uncased'
plm_path='../../pretrained_models/bert/bert-large-uncased'
for dataset_name in ${dataset_name_list[@]}
do
  for seed in ${seed_list[@]}
  do
    python train_plm.py --HP_hidden_dim=${hidden_size} --HP_lr=${lr} --HP_dropout=${dropout} --HP_batch_size=${batchsize} --HP_iteration=${iteration} --device=${device} --dataset_name=${dataset_name} --high_level_transformer=${plm_path} --model_dir=exps/${dataset_name}/m_${model_name}_s_${seed}_t_${time}/ --seed=${seed} > exp_log/d_${dataset_name}_m_${model_name}_s_${seed}_t_${time}.log
  done
done
#
model_name='bart-large'
plm_path='../../pretrained_models/bart/bart-large'
for dataset_name in ${dataset_name_list[@]}
do
  for seed in ${seed_list[@]}
  do
    python train_plm.py --HP_hidden_dim=${hidden_size} --HP_lr=${lr} --HP_dropout=${dropout} --HP_batch_size=${batchsize} --HP_iteration=${iteration} --device=${device} --dataset_name=${dataset_name} --high_level_transformer=${plm_path} --model_dir=exps/${dataset_name}/m_${model_name}_s_${seed}_t_${time}/ --seed=${seed} > exp_log/d_${dataset_name}_m_${model_name}_s_${seed}_t_${time}.log
  done
done
#
model_name='roberta-large'
plm_path='../../pretrained_models/roberta/roberta-large'
for dataset_name in ${dataset_name_list[@]}
do
  for seed in ${seed_list[@]}
  do
    python train_plm.py --HP_hidden_dim=${hidden_size} --HP_lr=${lr} --HP_dropout=${dropout} --HP_batch_size=${batchsize} --HP_iteration=${iteration} --device=${device} --dataset_name=${dataset_name} --high_level_transformer=${plm_path} --model_dir=exps/${dataset_name}/m_${model_name}_s_${seed}_t_${time}/ --seed=${seed} > exp_log/d_${dataset_name}_m_${model_name}_s_${seed}_t_${time}.log
  done
done