import torch
import torch.nn as nn
from IPython.core.debugger import set_trace

class MANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MANN, self).__init__()
        self.num_classes    =   output_size
        self.layer1 =   nn.LSTM(input_size=input_size + self.num_classes, hidden_size=hidden_size, batch_first=True)
        self.layer2 =   nn.LSTM(input_size=hidden_size, hidden_size=output_size, batch_first=True)


    def forward(self, input_images, input_lab):
        #set_trace()

        input_labels    =   input_lab.clone()
        input_labels[:, -1, :, :] = 0.0
        _inputs =   torch.cat((input_images, input_labels), dim=-1)

        over_class  =   []
        for iclass in range(self.num_classes):
            _input  =   _inputs[:, :, iclass, :]
            x   =   self.layer1(_input)[0]
            x   =   self.layer2(x)[0]
            over_class.append(x.unsqueeze(2))

        x = torch.cat(over_class, dim=2)
        return x


