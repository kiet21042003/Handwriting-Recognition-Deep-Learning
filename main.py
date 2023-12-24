from options import parser
from train import train_baseline
from train_wandb import train_baseline as train_wandb
                

if __name__=='__main__':
    
    args = parser.parse_args()
    if args.wandb:
        train_wandb(args)
    else:
        train_baseline(args)