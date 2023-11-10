import torch
from torchmetrics.text import CharErrorRate
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import tqdm

from data.dataset import HWDataset
from data.utils import collate_fn
from models.trba import TrBA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ALPHABET = "AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz.,-; '+*%^`=1234567890_\":?!~()[]#$%&/\\"

class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self):
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        character = ALPHABET
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character
        self.num_classes = len(self.character)

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i


    def encode(self, text, batch_max_length):
        """ 
        Convert text-label into text-index.

        Arguments:
        ----------
        text: 
            text labels of each image. [batch_size]
            
        batch_max_length: 
            max length of text label in the batch. 25 by default

        Returns:
        -------
        text : 
            the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
            text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            
        length : 
            the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """

        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.

        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts

def write_train_log(text, log_file, mode):
    with open(log_file, mode) as f:
        f.write(f"{text}\n")
     
def calc_cer(preds_str, labels, batch_size):
    cer = CharErrorRate()
    total_cer = 0
    for gt, pred in zip(labels, preds_str):
        pred_EOS = pred.find('[s]')
        gt = gt[:gt.find('[s]')]
        pred = pred[:pred_EOS]

    total_cer += cer([pred], [gt])
    val_cer = total_cer / batch_size
    return val_cer

def initialize(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("="*25, f"Configuration", "="*25)
    
    lr = args.lr
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    decay_rate = args.decay_rate
    save_every = args.save_every
    out_dir = os.path.join(args.out_dir, args.task)
    root_dir = args.root_dir
    
    print(f"[INFO] Device found: {device}")
    print(f"[INFO] Num of training epochs: {num_epochs}")
    print(f"[INFO] Batch size: {batch_size}")
    print(f"[INFO] Learning rate: {lr}")
    print(f"[INFO] Decaying rate: {decay_rate}")
    print(f"[INFO] Learning rate step every {args.lr_step_every} steps")
    print(f"[INFO] Save model every {save_every} epochs")
    print(f"[INFO] Using TPS: {args.stn_on}")
    print(f"[INFO] Max size {args.img_width} - Min size {args.img_height}")
    print(f"[INFO] Root data dir {root_dir}")
    print("="*65)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_log = os.path.join(out_dir, "logging.txt")
    write_train_log("Training Logging", out_log, 'w')

    img_channel = args.img_channel
    img_height = args.img_height
    img_width = args.img_width

    converter = AttnLabelConverter()
    model = TrBA(
        img_channel=img_channel,
        img_height=img_height,
        img_width=img_width,
        num_class=converter.num_classes,
        max_len=args.max_length,
        stn_on=args.stn_on
    ).to(device)
    
    if args.weights != '':
        print(f"[INFO] Load weights from {args.weights}")
        model.load_state_dict(torch.load(args.weights, map_location=torch.device(device)))
    
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=args.lr_step_every, gamma=decay_rate)

    trainset = HWDataset(root_dir, min_size=img_width, max_size=img_height, mode='train', max_len=args.max_length)
    testset = HWDataset(root_dir, min_size=img_width, max_size=img_height, mode='test', max_len=args.max_length)
    
    print(f"[INFO] Training data size: {len(trainset)}")
    print(f"[INFO] Validation data size: {len(testset)}")
    

    # DataLoader
    train_loader = DataLoader(trainset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4, shuffle=False)
    
    return {
        'converter': converter,
        'model': model,
        'criterion': criterion,
        'scheduler': scheduler,
        'optimizer': optimizer,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'out_log': out_log,
        'out_dir': out_dir
    }
    
def initialize_for_infer(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    img_channel = args.img_channel
    img_height = args.img_height
    img_width = args.img_width

    converter = AttnLabelConverter()
    model = TrBA(
        img_channel=img_channel,
        img_height=img_height,
        img_width=img_width,
        num_class=converter.num_classes,
        max_len=args.max_length,
        stn_on=args.stn_on
    ).to(device)
 
    if args.weights != '':
        print(f"[INFO] Load weights from {args.weights}")
        model.load_state_dict(torch.load(args.weights, map_location=torch.device(device)))   
    
    return {
        'converter': converter,
        'model': model
    }