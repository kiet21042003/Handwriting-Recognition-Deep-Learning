import os
import torch
import numpy as np
from tqdm import tqdm
from options import parser

from utils import *

def test(args):
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    configs = initialize(args)
    model = configs['model']
    criterion = configs['criterion']
    converter = configs['converter']
    out_log = configs['out_log']

    model.eval()
    loss_avg = []
    val_cer = []
    length_for_pred = torch.IntTensor([args.max_length] * args.batch_size)
    text_for_pred = torch.LongTensor(args.batch_size, args.max_length + 1).fill_(0)


    print("Start evaluating")
    for images, labels in tqdm(configs['test_loader'], total=len(configs['test_loader'])):
        text, length = converter.encode(labels, batch_max_length=args.max_length)
        
        with torch.no_grad():
            preds, visual_feature = model(images.to(device), text[:, :-1], True) # Align with Attention
            
        try:
            target = text[:, 1:]
            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
            loss_avg.append(cost.detach().cpu())
            
            _, preds_index = preds.detach().cpu().max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(target, length)
            val_cer.append(calc_cer(preds_str, labels, args.batch_size))
        except:
            pass
        
    val_cer = np.mean(np.array(val_cer))
        
    loss_avg = torch.stack(loss_avg, 0)
    loss_avg = loss_avg.cpu().mean()
    
    log_message = f"[INFO] Validation Loss {loss_avg} | CER {val_cer}"
    write_train_log(log_message, out_log, 'a')
    print(log_message)

if __name__=='__main__':
    args = parser.parse_args()
    test(args)