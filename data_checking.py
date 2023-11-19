import torch
from data.dataset import HWDataset
from tqdm import tqdm


def check(root_dir):
    trainset = HWDataset(root_dir, min_size=30, max_size=300, mode='train', max_len=25)
    testset = HWDataset(root_dir, min_size=30, max_size=300, mode='test', max_len=25)
    
    longest = 0
    for i in tqdm(range(len(trainset)), total=len(trainset)):
        try:
            x, y, _, _ = trainset[i]
            if len(y) > longest:
                longest = len(y)
        except:
            print("Exception:", i, "at", trainset.get_item_name(i))
    print("Train longest:", longest)
            
    longest = 0
    for i in tqdm(range(len(testset)), total=len(testset)):
        try:
            x, y, _, _ = testset[i]
            if len(y) > longest:
                longest = len(y)
                print(y, testset.get_item_name(i))
        except:
            print("Exception:", i, "at", testset.get_item_name(i))
            
    print("Test longest:", longest)
            
root_dir = '/mnt/disk1/nmduong/hust/intro2dl/data/words'
check(root_dir)

