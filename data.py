import torch
import torchvision
import matplotlib.pyplot as plt


from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

from torchvision.utils import make_grid

def greb_data(data_dir):
    trainset = torchvision.datasets.ImageFolder(data_dir+"/train",
                                               transform = torchvision.transforms.Compose([torchvision.transforms.Resize((150,150)),
                                                                                           torchvision.transforms.ToTensor()]))

    testset = torchvision.datasets.ImageFolder(data_dir+"/test",
                                               transform = torchvision.transforms.Compose([torchvision.transforms.Resize((150,150)),
                                                                                           torchvision.transforms.ToTensor()]))
    return trainset, testset


def check_data_classes(data_dir):
    print("classes : \n", greb_data(data_dir)[0].classes)
    
def check_image(img,label):
    plt.imshow(img.permute(1,2,0))
    plt.title(label)

    
def trainset_preparation(data_dir, batch_size, val_size, verbose=False):

    train_size = len(greb_data(data_dir)[0]) - val_size 

    train_data,val_data = random_split(greb_data(data_dir)[0],[train_size,val_size])
    
    if verbose:
        print(f"Length of Train Data : {len(train_data)}")
        print(f"Length of Validation Data : {len(val_data)}")

    #output
    #Length of Train Data : 12034
    #Length of Validation Data : 2000

    #load the train and validation into batches.
    train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    val_dl = DataLoader(val_data, batch_size*2, num_workers = 4, pin_memory = True)
    return train_dl, val_dl



def show_batch(dl):
    """Plot images grid of single batch"""
    for images, labels in dl:
        fig,ax = plt.subplots(figsize = (16,12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=16).permute(1,2,0))
        break
        
