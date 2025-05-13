import argparse


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,default=r'/hpc/data/home/bme/zhangjd1/nBME/ALL_DATA')
    parser.add_argument('--csv_path', type=str,default=r'/hpc/data/home/bme/zhangjd1/nBME/ALL_DATA/Clinics.csv')
    parser.add_argument('--clinical_path', type=str, default=r'/hpc/data/home/bme/zhangjd1/nBME/ALL_DATA/Clinics.csv')
    parser.add_argument('--model_name', type=str, default='Ex1')
    parser.add_argument('--model_path', type=str, default='./checkpoints')
    parser.add_argument('--writer_comment', type=str, default='Results')
    parser.add_argument('--save_model', type=bool, default=True)

    # MODEL PARAMETER
    parser.add_argument('--img_size_US', type=int, default=256)
    parser.add_argument('--img_size_MM', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--log_step', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)

    parser.add_argument('--loss_function', type=str, default='CE')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step'])
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--warmup_decay', type=float, default=0.01)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--step', type=int, default=5)


    config = parser.parse_args()
    return config