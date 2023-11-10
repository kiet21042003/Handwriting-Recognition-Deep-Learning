import os
import torch
import numpy as np
from tqdm import tqdm
import wandb

from utils import *
    
def train(args):
    configs = initialize(args)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    model = configs['model']
    criterion = configs['criterion']
    optimizer = configs['optimizer']
    converter = configs['converter']
    scheduler = configs['scheduler']
    out_log = configs['out_log']
    out_dir = configs['out_dir']
    train_loader = configs['train_loader']
    test_loader = configs['test_loader']

    best_cer = 1e5
    best_loss = 1e5
    
    model.to(device)
    global_step = 0
    
    for epoch in range(args.num_epochs):
        total_steps = len(train_loader)
        current_step = 0
        
        # Training
        for images, labels in tqdm(train_loader, total=total_steps):
            current_step += 1
            global_step += 1
            batch_size = images.size(0)
            
            # Val step
            if global_step % args.val_every == 0:
                print("[TEST] Validation step")
                model.eval()
                loss_avg = []
                val_cer = []
                length_for_pred = torch.IntTensor([args.max_length] * batch_size)
                text_for_pred = torch.LongTensor(batch_size, args.max_length + 1).fill_(0)

                for images, labels in tqdm(test_loader, total=len(test_loader)):
                    text, length = converter.encode(labels, batch_max_length=args.max_length)
                    
                    with torch.no_grad():
                        preds, visual_feature = model(images.to(device), text[:, :-1], True) # Align with Attention
                    target = text[:, 1:]
                    cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
                    loss_avg.append(cost.detach().cpu())

                    try:
                        _, preds_index = preds.detach().cpu().max(2)
                        preds_str = converter.decode(preds_index, length_for_pred)
                        labels = converter.decode(target, length)
                        val_cer.append(calc_cer(preds_str, labels, batch_size))
                    except: 
                        pass

                val_cer = np.mean(np.array(val_cer))
                    
                loss_avg = torch.stack(loss_avg, 0)
                loss_avg = loss_avg.cpu().mean()
                
                log_message = f"[INFO] [{epoch}/{args.num_epochs}|{current_step}/{total_steps}] Validation Loss {loss_avg} | CER {val_cer}"
                write_train_log(log_message, out_log, 'a')
                print(log_message)
                
                # Best loss
                if loss_avg < best_loss:
                    best_loss = loss_avg
                    model.train()
                    print(f"[INFO] Saving model with best val loss {best_loss}")
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, os.path.join(out_dir, f"best_{best_loss}.pth"))    
                
            # Backward
            text, length = converter.encode(labels, batch_max_length=args.max_length)

            model.train()
            optimizer.zero_grad()            
            
            preds, visual_feature = model(images.to(device), text[:, :-1], True) # align with Attention forward
            targets = text[:, 1:] # remove [GO] symbol
            cost = criterion(preds.view(-1, preds.shape[-1]), targets.contiguous().view(-1))
            
            cost.backward()
            optimizer.step()
            scheduler.step()      
            
            # Log step
            if current_step % args.log_every == 0:
                log_message = f"[INFO] [{epoch}/{args.num_epochs}|{current_step}/{total_steps}] Train Loss {cost.detach().cpu().mean()}"
                write_train_log(log_message, out_log, 'a')
                
                
        # if (epoch+1) % args.save_every == 0:
        #     model.train()
        #     print(f"[INFO] Saving model at epoch {epoch+1}")
        #     torch.save(model.state_dict(), os.path.join(args.out_dir, f"E_{epoch+1}.pth"))
            
            
        torch.cuda.empty_cache()