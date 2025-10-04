#dataset_list=('clinical' 'wnut16' 'wnut17')
dataset_list=('clinical')
encoder_decoder_name_list=('../../pretrained_models/bart/bart-base')
num_train_epochs=20
train_batch_size=70
device=0
seed=42
#max_seq_length 可以尝试50、100、200
#max_length 可以尝试25、50
max_seq_length=50
max_length=25
for dataset_name in ${dataset_list[@]}
do
  for encoder_decoder_name in ${encoder_decoder_name_list[@]}
  do
    python train_inference.py --train_batch_size=${train_batch_size} --num_train_epochs=${num_train_epochs} --max_seq_length=${max_seq_length} --max_length=${max_length} --dataset_name=${dataset_name} --encoder_decoder_name=${encoder_decoder_name} --device=${device}
  done
done