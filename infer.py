import torch
from models.trba import TrBA
from options import parser
from utils import initialize_for_infer
from data.utils import resize_image, padding
import cv2
import os
import numpy as np

if __name__=='__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    args = parser.parse_args()
    configs = initialize_for_infer(args)
    model = configs['model']
    converter = configs['converter']
    
    test_dir = "./images"
    test_fn = [os.path.join(test_dir, f) for f in os.listdir(test_dir)]

    batch_size = 1
    
    for fn in test_fn:
        image = cv2.imread(fn).astype(np.float32)
        if image.ndim==2:
            image = image[np.newaxis]
        image = image.transpose((2,0,1))
        image = torch.from_numpy(image).type(torch.FloatTensor)
        image = resize_image(image, min_size=args.img_height, max_size=args.img_width)
        image = padding(image, min_size=args.img_height, max_size=args.img_width)
        
        length_for_pred = torch.IntTensor([args.max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, args.max_length + 1).fill_(0).to(device)
        
        image = image.unsqueeze(0)
        model.eval()
        
        with torch.no_grad():
            preds, _ = model(image.to(device), text_for_pred, is_train=False)
        _, preds_index = preds.max(2)
        pred_str = converter.decode(preds_index, length_for_pred)[0]
        # print(pred_str)
        
        pred_EOS = pred_str.find('[s]')
        pred_str = pred_str[:pred_EOS]
        
        print(fn, pred_str)
        