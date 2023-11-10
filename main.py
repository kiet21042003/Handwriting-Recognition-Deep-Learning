import torch
from options import parser
from train import train
                

if __name__=='__main__':
    
    args = parser.parse_args()
    train(args)