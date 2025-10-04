dataset_name='clinical'
lr=1.0
dropout=0.5
iteration=100
batchsize=16
device=0
time=$(date "+%Y%m%d-%H%M%S")
model_name='wlstm_nochar'
seed=22
python train_lstm.py --HP_lr=${lr} --HP_dropout=${dropout} --HP_batch_size=${batchsize} --HP_iteration=${iteration} --device=${device} --dataset_name=${dataset_name} --model_dir=exps/${dataset_name}/m_${model_name}_s_${seed}_t_${time}/ --seed=${seed} > exp_log/d_${dataset_name}_m_${model_name}_s_${seed}_t_${time}.log &
seed=32
python train_lstm.py --HP_lr=${lr} --HP_dropout=${dropout} --HP_batch_size=${batchsize} --HP_iteration=${iteration} --device=${device} --dataset_name=${dataset_name} --model_dir=exps/${dataset_name}/m_${model_name}_s_${seed}_t_${time}/ --seed=${seed} > exp_log/d_${dataset_name}_m_${model_name}_s_${seed}_t_${time}.log &
seed=42
python train_lstm.py --HP_lr=${lr} --HP_dropout=${dropout} --HP_batch_size=${batchsize} --HP_iteration=${iteration} --device=${device} --dataset_name=${dataset_name} --model_dir=exps/${dataset_name}/m_${model_name}_s_${seed}_t_${time}/ --seed=${seed} > exp_log/d_${dataset_name}_m_${model_name}_s_${seed}_t_${time}.log &
seed=52
python train_lstm.py --HP_lr=${lr} --HP_dropout=${dropout} --HP_batch_size=${batchsize} --HP_iteration=${iteration} --device=${device} --dataset_name=${dataset_name} --model_dir=exps/${dataset_name}/m_${model_name}_s_${seed}_t_${time}/ --seed=${seed} > exp_log/d_${dataset_name}_m_${model_name}_s_${seed}_t_${time}.log &
seed=62
python train_lstm.py --HP_lr=${lr} --HP_dropout=${dropout} --HP_batch_size=${batchsize} --HP_iteration=${iteration} --device=${device} --dataset_name=${dataset_name} --model_dir=exps/${dataset_name}/m_${model_name}_s_${seed}_t_${time}/ --seed=${seed} > exp_log/d_${dataset_name}_m_${model_name}_s_${seed}_t_${time}.log &
