dataset_name_list=('clinical')
lr=0.015
dropout=0.5
iteration=100
batchsize=16
seed_list=(22 32 42 52 62)
device=0
time=$(date "+%Y%m%d-%H%M%S")
#seed_list=(42)
model_name='wlstm_nochar'
for dataset_name in ${dataset_name_list[@]}
do
  for seed in ${seed_list[@]}
  do
    python train_lstm.py --HP_lr=${lr} --HP_dropout=${dropout} --HP_batch_size=${batchsize} --HP_iteration=${iteration} --device=${device} --dataset_name=${dataset_name} --model_dir=exps/${dataset_name}/m_${model_name}_s_${seed}_t_${time}/ --seed=${seed} > exp_log/d_${dataset_name}_m_${model_name}_s_${seed}_t_${time}.log
  done
done
#
model_name='wlstm_clstm'
for dataset_name in ${dataset_name_list[@]}
do
  for seed in ${seed_list[@]}
  do
    python train_lstm.py  --use_char --char_feature_extractor='LSTM' --HP_lr=${lr} --HP_dropout=${dropout} --HP_batch_size=${batchsize} --HP_iteration=${iteration} --device=${device} --dataset_name=${dataset_name} --model_dir=exps/${dataset_name}/m_${model_name}_s_${seed}_t_${time}/ --seed=${seed} > exp_log/d_${dataset_name}_m_${model_name}_s_${seed}_t_${time}.log
  done
done
model_name='wlstm_ccnn'
for dataset_name in ${dataset_name_list[@]}
do
  for seed in ${seed_list[@]}
  do
    python train_lstm.py  --use_char --char_feature_extractor='CNN' --HP_lr=${lr} --HP_dropout=${dropout} --HP_batch_size=${batchsize} --HP_iteration=${iteration} --device=${device} --dataset_name=${dataset_name} --model_dir=exps/${dataset_name}/m_${model_name}_s_${seed}_t_${time}/ --seed=${seed} > exp_log/d_${dataset_name}_m_${model_name}_s_${seed}_t_${time}.log
  done
done
model_name='wlstm_crf_ccnn'
for dataset_name in ${dataset_name_list[@]}
do
  for seed in ${seed_list[@]}
  do
    python train_lstm.py --use_crf --use_char --char_feature_extractor='CNN' --HP_lr=${lr} --HP_dropout=${dropout} --HP_batch_size=${batchsize} --HP_iteration=${iteration} --device=${device} --dataset_name=${dataset_name} --model_dir=exps/${dataset_name}/m_${model_name}_s_${seed}_t_${time}/ --seed=${seed} > exp_log/d_${dataset_name}_m_${model_name}_s_${seed}_t_${time}.log
  done
done