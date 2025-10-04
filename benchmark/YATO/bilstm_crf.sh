dataset_name='COVID-CETS'
#seed_list=(22 32 42 52 62)
seed_list=(42)
model_name='wordlstm_crf_nochar'
for seed in ${seed_list[@]}
do
  python train_lstm.py --model_dir=exps/${dataset_name}/m_${model_name}_s_${seed}/ --seed=${seed} > exp_log/d_${dataset_name}_m_${model_name}_s_${seed}.log &
done
# CUDA_VISIBLE_DEVICES=1
model_name='wordlstm_crf_charlstm'
for seed in ${seed_list[@]}
do
  python train_lstm.py --use_char --char_feature_extractor='LSTM' --model_dir=exps/${dataset_name}/m_${model_name}_s_${seed}/ --seed=${seed} > exp_log/d_${dataset_name}_m_${model_name}_s_${seed}.log &
done
model_name='wordlstm_crf_charcnn'
for seed in ${seed_list[@]}
do
  python train_lstm.py --use_char --char_feature_extractor='CNN' --model_dir=exps/${dataset_name}/m_${model_name}_s_${seed}/ --seed=${seed} > exp_log/d_${dataset_name}_m_${model_name}_s_${seed}.log &
done