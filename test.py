import os
import torch
import numpy as np
from tqdm import tqdm
import wandb

from utils import *
    
def test_baseline(args):
    configs = initialize_for_baseline(args)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    model = configs['model']
    criterion = configs['criterion']
    converter = configs['converter']
    test_loader = configs['test_loader']
    batch_size = configs['batch_size']
    
    model.to(device)
    
    print("[TEST] Validation step")
    model.eval()
    loss_avg = []
    val_cer = []
    length_for_pred = torch.IntTensor([args.max_length] * batch_size).to(device)
    text_for_pred = torch.LongTensor(batch_size, args.max_length + 1).fill_(0).to(device)

    for images, labels in tqdm(test_loader, total=len(test_loader)):
        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=args.max_length)
        
        with torch.no_grad():
            preds, visual_feature = model(images.to(device), text_for_pred, False) # Align with Attention
            
        preds = preds[:, :text_for_loss.shape[1]-1, :]
        target = text_for_loss[:, 1:]
        cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
        loss_avg.append(cost.detach().cpu())

        try:
            _, preds_index = preds.detach().cpu().max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(target, length_for_loss)
            val_cer.append(calc_cer(preds_str, labels, batch_size))
        except: 
            pass

    val_cer = np.mean(np.array(val_cer))
        
    loss_avg = torch.stack(loss_avg, 0)
    loss_avg = loss_avg.cpu().mean()
    
    print(f"Validation CER: {val_cer} - Validation Loss: {loss_avg}")
    

                

if __name__=='__main__':
    from options import parser  
    args = parser.parse_args()
    test_baseline(args)