import os
import torch
import numpy as np
from tqdm import tqdm
import wandb

from utils import *
    
def train_baseline(args):
    configs = initialize_for_baseline(args)
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
    
    wandb.init(
        project='handwriting-ocr',
        entity='aiotlab',
        name=args.task
    )
    
    for epoch in range(args.num_epochs):
        total_steps = len(train_loader)
        current_step = 0
        train_loss_epoch = 0
        
        # Training
        for images, labels in tqdm(train_loader, total=total_steps):
            track_dict = dict()             
            current_step += 1
            global_step += 1
            batch_size = images.size(0)
            
            # Val step
            if global_step % args.val_every == 0:
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
                
                track_dict['cer'] = val_cer
                track_dict['val_loss'] = loss_avg
                
                log_message = f"[INFO] [{epoch}/{args.num_epochs}|{current_step}/{total_steps}] Validation Loss {loss_avg} | CER {val_cer}"
                write_train_log(log_message, out_log, 'a')
                print(log_message)
                
                # Best loss
                if val_cer < best_cer:
                    best_cer = val_cer
                    model.train()
                    print(f"[INFO] Saving model with best val cer {best_cer}")
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, os.path.join(out_dir, f"best_cer_{best_cer}.pth"))    
                    track_dict['best_cer'] = best_cer
                    
                if loss_avg < best_loss:
                    best_loss = loss_avg
                    model.train()
                    track_dict['best_loss'] = best_loss
                
            # Backward
            text, length = converter.encode(labels, batch_max_length=args.max_length)

            model.train()
            optimizer.zero_grad()            
            
            preds, visual_feature = model(images.to(device), text[:, :-1], True) # align with Attention forward
            targets = text[:, 1:] # remove [GO] symbol
            loss = criterion(preds.view(-1, preds.shape[-1]), targets.contiguous().view(-1))
            
            loss.backward()
            optimizer.step()     
            
            # Log step
            if current_step % args.log_every == 0:
                log_message = f"[INFO] [{epoch}/{args.num_epochs}|{current_step}/{total_steps}] Train Loss {loss.detach().cpu().mean()}"
                write_train_log(log_message, out_log, 'a')
            train_loss_epoch += loss.detach().cpu().item()
            track_dict['train_loss'] = loss.detach().cpu().item()
            wandb.log(track_dict)
        
        scheduler.step(train_loss_epoch / total_steps)
        wandb.log({'train_loss_epoch': train_loss_epoch / total_steps,
                   'lr': optimizer.param_groups[0]['lr']})
        
                
        # if (epoch+1) % args.save_every == 0:
        #     model.train()
        #     print(f"[INFO] Saving model at epoch {epoch+1}")
        #     torch.save(model.state_dict(), os.path.join(args.out_dir, f"E_{epoch+1}.pth"))
            
            
        torch.cuda.empty_cache()