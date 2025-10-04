time=$(date "+%Y%m%d-%H%M%S")
model_name='covid-twitter-bert-v2'
exp_dir='./exps/clinical/m_covid-twitter-bert-v2_s_22_t_20220606-005037/'
python decode.py --exp_dir=${exp_dir} > exp_log/decode_d_${model_name}_t_${time}.log