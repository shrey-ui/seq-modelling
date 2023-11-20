import torch
import torch.nn as nn
import torchvision
import numpy as np 
import os
import torch.nn as nn
import utils
import torchviz
from torchviz import make_dot
class CompModel(nn.Module):
    def __init__(self, dimension):
        super(CompModel, self).__init__()
        self.dim= dimension
        self.layer= nn.Linear(2048, self.dim)

    def forward(self, feat_vector):
        return self.layer(feat_vector)

class RNN_Model(nn.Module):
    def __init__(self, hidden_size, input_size, output_size):
        super(RNN_Model, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.inp1= nn.Linear(self.input_size + self.hidden_size, 2*self.input_size)
        self.inp2= nn.Linear(2*self.input_size, output_size)
        self.softmax= nn.LogSoftmax(dim= 1)
        self.hidden_layer = nn.Linear(self.input_size+self.hidden_size, self.hidden_size)
        
        #self.comp= CompModel(self.hidden_siz
        
    def forward(self, input_wordseq, hidden_input):
        
    
        #hidden_input= self.comp(hidden_input)
        #print(input_wordseq.shape)
        #print(hidden_input.shape)
        combined= torch.cat([input_wordseq, hidden_input], dim= 1)
        #print(combined.shape)
        
        out1= self.inp1(combined)
        #print(out1.shape)
        out2= self.inp2(out1)
        #print(out2.shape)
        output= self.softmax(out2)
        #print(output.shape)
        #output= self.softmax(self.inp2(self.inp1(combined)))        
        hidden_input= self.hidden_layer(combined)
        return output, hidden_input

def train(train_dl, vocab_size, max_len):
    model= RNN_Model(2048, max_len, vocab_size)
    criterion= nn.NLLLoss()
    lr= 0.001
    optimizer= torch.optim.SGD(model.parameters(), lr= lr)
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(1):
        epoch_loss= 0

        for ind, sample in enumerate(train_dl):
            input_rnn, output_rnn= sample
            image_feat = input_rnn[0]
            input_seq= input_rnn[1]

            #hidden_input = image_feat
            #hidden_input= torch.Tensor(hidden_input).squeeze(0)
            if((ind+1)%1000 == 0):
                print(f"EPOCH : {epoch + 1} Image : {ind + 1} : Total Loss : {epoch_loss/(ind+1)}")

            for caption in range(5):
                loss = torch.Tensor([0])
                hidden_input= image_feat
                hidden_input= torch.Tensor(hidden_input).squeeze(0)
                for input_ind in range(len(input_seq[caption])):
                    
                    act_input= input_seq[caption][input_ind]
                    act_output= output_rnn[caption][input_ind]
                    act_input= act_input/vocab_size 
                     
                    #hidden_input= image_feat
                    #hidden_input= torch.Tensor(hidden_input).squeeze(0)
                    output, hidden_input= model(act_input, hidden_input)
                    #print(output.shape)
                    #print(act_output.shape)
                    act_output= torch.argmax(act_output, dim= 1)
                    print(output.shape)
                    print(act_output.shape)
            

                    #print(act_output.shape)
                    l= criterion(output, act_output)
                    loss+= l
                

                optimizer.zero_grad()
                
                loss.backward(retain_graph= False)
                optimizer.step()
                #for p in model.parameters():
                #    p.data.add_(p.grad.data, alpha=-lr)
                epoch_loss= 0
                #epoch_loss+= loss.item()
        
        print(f"Epoch : {epoch+ 1} ---- LOSS : {epoch_loss/(ind+1)}")

if __name__ == "__main__":
    
    
    train_dl, max_len, vocab_size= utils.get_all_required_data()
    

    train(train_dl, vocab_size, max_len)



