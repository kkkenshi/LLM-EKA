nohup bash run_bilstm_00.sh &
nohup bash run_bilstm_01.sh &
nohup bash run_bilstm_10.sh &
nohup bash run_bilstm_11.sh &
nohup bash run_bilstm_20.sh &
nohup bash run_bilstm_21.sh &
#dataset_name_list=('clinical' 'wnut16' 'wnut17')
#lr=0.015
#dropout=0.5
#iteration=100
#batchsize=16
#seed_list=(22 32 42 52 62)
#device=0
#time=$(date "+%Y%m%d-%H%M%S")
##seed_list=(42)
#model_name='wordlstm_crf_nochar'
#for dataset_name in ${dataset_name_list[@]}
#do
#  for seed in ${seed_list[@]}
#  do
#    python train_lstm.py --HP_lr=${lr} --HP_dropout=${dropout} --HP_batch_size=${batchsize} --HP_iteration=${iteration} --device=${device} --dataset_name=${dataset_name} --model_dir=exps/${dataset_name}/m_${model_name}_s_${seed}_t_${time}/ --seed=${seed} > exp_log/d_${dataset_name}_m_${model_name}_s_${seed}_t_${time}.log
#  done
#done
##
#model_name='wordlstm_crf_lstmchar'
#for dataset_name in ${dataset_name_list[@]}
#do
#  for seed in ${seed_list[@]}
#  do
#    python train_lstm.py  --use_char --char_feature_extractor='LSTM' --HP_lr=${lr} --HP_dropout=${dropout} --HP_batch_size=${batchsize} --HP_iteration=${iteration} --device=${device} --dataset_name=${dataset_name} --model_dir=exps/${dataset_name}/m_${model_name}_s_${seed}_t_${time}/ --seed=${seed} > exp_log/d_${dataset_name}_m_${model_name}_s_${seed}_t_${time}.log
#  done
#done