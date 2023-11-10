import torch
from data.dataset import HWDataset
from tqdm import tqdm


def check(root_dir):
    trainset = HWDataset(root_dir, min_size=30, max_size=300, mode='train', max_len=25)
    testset = HWDataset(root_dir, min_size=30, max_size=300, mode='test', max_len=25)
    
    for i in tqdm(range(len(trainset)), total=len(trainset)):
        try:
            x, y, _, _ = trainset[i]
        except:
            print("Exception:", i, "at", trainset.get_item_name(i))
            
    for i in tqdm(range(len(testset)), total=len(testset)):
        try:
            x, y = testset[i]
        except:
            print("Exception:", i, "at", testset.get_item_name(i))
            
root_dir = '/mnt/disk1/nmduong/hust/intro2dl/data/words'
check(root_dir)

