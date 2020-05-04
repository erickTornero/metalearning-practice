import numpy as np
import random
from load_data import DataGenerator
from model import MANN
import torch
import torch.nn as nn
import torch.optim as optim
from IPython.core.debugger import set_trace

loss_function   =   nn.CrossEntropyLoss()
def compute_loss(predicted, labels):
    total_loss  =   0.0
    for iclass in range(labels.shape[-1]):
        _preds  =   predicted[:, -1, iclass, :]
        _labls  =   labels[:, -1, iclass, :].argmax(1)

        loss    =   loss_function(_preds, _labls)
        total_loss +=loss

    return total_loss


number_classes  =   5
number_samples_xclass    =   10
batch_size      =   32
image_size      =   784
hidden_units    =   128
learning_rate   =   1e-3
device          =   torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model           =   MANN(image_size, hidden_units, number_classes).to(device)
optimizer       =   optim.Adam(lr=learning_rate, params=model.parameters())

print('Use device:> ', device)
def train():
    datagenerator   =   DataGenerator(number_classes, number_samples_xclass)
    print('Start training')
    for step in range(50000):
        _imgs, _labels  =   datagenerator.sample_batch('train', batch_size)

        _imgs_tensor, _labels_tensor    =   torch.tensor(_imgs, dtype=torch.float32, device=device), torch.tensor(_labels, dtype=torch.float32, device=device)

        output  =   model(_imgs_tensor, _labels_tensor)
        optimizer.zero_grad()
        loss    =   compute_loss(output, _labels_tensor)
        
        loss.backward()        
        optimizer.step()

        if step%50 == 0:
            set_trace()
            _imgs, _labels = datagenerator.sample_batch('test', 100)
            _imgs_tensor, _labels_tensor = torch.tensor(_imgs, dtype=torch.float32, device=device), torch.tensor(_labels, dtype=torch.float32, device=device)
            with torch.no_grad():
                output_t    =   model(_imgs_tensor, _labels_tensor)
                pred_lbls   =   np.asarray(output_t[:, -1, :,:].argmax(2).to('cpu'))
                _labels_tn  =   _labels[:,-1,:,:].argmax(2)
                accuracy    =   (_labels_tn == pred_lbls).mean()
                print('accuracy ->\t{}'.format(accuracy))

if __name__ == "__main__":
    train()