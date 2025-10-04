dataset_name='clinical'
encoder_decoder_name='../../pretrained_models/bart/bart-base'
num_train_epochs=1
train_batch_size=70
device=0
seed=42
max_seq_length=50
max_length=25
python train_inference.py --train_batch_size=${train_batch_size} --num_train_epochs=${num_train_epochs} --max_seq_length=${max_seq_length} --max_length=${max_length} --dataset_name=${dataset_name} --encoder_decoder_name=${encoder_decoder_name} --device=${device}