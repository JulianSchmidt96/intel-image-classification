
import torch
import data
from Classifier import Classifier

 
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

    
def fit(model, epochs, lr, train_loader, val_loader, opt_func = torch.optim.Adam):
    history = []
    optimizer = opt_func(model.parameters(),lr)
    for epoch in range(epochs):
        print(epoch)
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
                
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
        
    return history

def plot_accuracies(history):
    """ Plot the history of accuracies"""
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');
    

if __name__ == "__main__":
    
    
    
    # Global constants
    EPOCHS = 10
    LR = 0.001
    BATCHSIZE = 100
    VALSIZE = 2000
    # Globals
    
    train_dl, val_dl = data.trainset_preparation("dataset", BATCHSIZE, VALSIZE)
    model = Classifier()
    
    fit(model,EPOCHS, LR, train_dl, val_dl)