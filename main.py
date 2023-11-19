from options import parser
from train import *
                

if __name__=='__main__':
    
    args = parser.parse_args()
    train_baseline(args)