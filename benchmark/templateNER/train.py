import pandas as pd
import logging
from seq2seq_model import Seq2SeqModel
import argparse
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
def get_params():
    args = argparse.ArgumentParser()
    ### I/O ###
    args.add_argument("--dataset_name", default='clinical', type=str)# could be clinical/wnut16/wnut17/conll
    args.add_argument("--output_dir", default='./exp/template', type=str)#实验存放地址 exps/dataset_name/模型+seed
    args.add_argument("--encoder_decoder_name", default="../../pretrained_models/bart/bart-base", type=str)#实验存放地址 exps/dataset_name/模型+seed

    args.add_argument("--num_train_epochs", default=20, type=int)
    args.add_argument("--train_batch_size", default=70, type=int)
    args.add_argument("--max_seq_length", default=50, type=int)
    args.add_argument("--max_length", default=25, type=int)
    args.add_argument("--num_train_epochs", default=20, type=int)
    args.add_argument("--device", default=0, type=int)
    args.add_argument("--seed", default=42, type=int)

    args = args.parse_args()

    dataset2path = {
        'clinical':('../../datasets/clinical_ner/7-1.5-1.5/train.csv','../../datasets/clinical_ner/7-1.5-1.5/dev.csv','../../datasets/clinical_ner/7-1.5-1.5/test.bioes'),
        'wnut16':('../../datasets/wnut16/train.csv','../../datasets/wnut16/dev.csv','../../datasets/wnut16/test.bioes'),
        'wnut17': (
        '../../datasets/wnut17/train.csv', '../../datasets/wnut17/dev.csv', '../../datasets/wnut17/test.bioes'),
        'conll':('./datasets/CoNLL03_NER/train.csv','./datasets/CoNLL03_NER/dev.csv','./datasets/CoNLL03_NER/test.bioes')
    }
    assert args.dataset_name in ['clinical','wnut16','wnut17','conll']
    args.train_dir,args.dev_dir,args.test_dir = dataset2path[args.dataset_name]

    params = {}
    for k, v in vars(args).items():
        if v=='':
            params[k] = None
        else:
            params[k] = v
    return params, args
if __name__ == '__main__':
    config_dict, args = get_params()
    train_data = pd.read_csv(args.train_dir, sep='\t').values.tolist()
    train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])

    eval_data = pd.read_csv(args.dev_dir, sep='\t').values.tolist()
    eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])

    model_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "max_seq_length": args.max_seq_length,
        "train_batch_size": args.train_batch_size,
        "num_train_epochs": args.num_train_epochs,
        "save_eval_checkpoints": False,
        "save_model_every_epoch": False,
        "evaluate_during_training": True,
        "evaluate_generated_text": True,
        "evaluate_during_training_verbose": True,
        "use_multiprocessing": False,
        "max_length": args.max_length,
        "manual_seed": args.seed,
        "save_steps": 11898,
        "gradient_accumulation_steps": 1,
        "output_dir": args.output_dir,
        "best_model_dir": args.output_dir + '/best_model'
    }
    print(model_args)
    # Initialize model
    model = Seq2SeqModel(
        encoder_decoder_type="bart",
        encoder_decoder_name=args.encoder_decoder_name,
        args=model_args,
        # use_cuda=False,
        cuda_device=args.device
    )


    # Train the model
    model.train_model(train_df, eval_data=eval_df)

    # Evaluate the model
    results = model.eval_model(eval_df)

    # Use the model for prediction

    print(model.predict(["Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a Group C championship match on Friday."]))
