import torch
import torch
import torch.nn as nn
import torchvision
import numpy as np 
import os
import torch.nn as nn
import utils
import torchviz
from torchviz import make_dot
from torch.autograd import Variable
import torchvision.transforms as T

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
        #self.embedding= nn.Embedding(output_size, self.hidden_size, scale_grad_by_freq= True)
        self.inp1= nn.Linear(self.output_size, self.hidden_size)
        #self.inp2= nn.Linear(2*self.input_size, 512)
      
        self.softmax= nn.LogSoftmax(dim= 1)
        self.hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.comp= nn.Linear(2048, self.hidden_size)
        self.output_lay = nn.Linear(2*hidden_size, output_size)
        
    def forward(self, input_wordseq, hidden_input):
       #print(input_wordseq.shape)
        if(hidden_input.shape[1]!= self.hidden_size):
          hidden_input= self.comp(hidden_input)
        #embed= self.embedding(input_wordseq)
        out1= self.inp1(input_wordseq)
       #print(out1.shape)
       # print(hidden_input.shape)
        combined= torch.cat([out1, hidden_input], dim= 1)
     
        output= self.softmax(self.output_lay(combined))
              
        hidden_input= self.hidden_layer(hidden_input)

        return output, hidden_input

class RNN_Model_(nn.Module):

  def __init__(self, hidden_size, input_size, output_size):
    super(RNN_Model_, self).__init__()
    self.hidden_size= hidden_size
    self.input_size= input_size
    self.output_size= output_size
    self.comp= nn.Linear(2048, self.hidden_size)
    self.output_layer= nn.Linear(self.hidden_size, self.output_size)
    self.rnn_layer= nn.RNN(input_size= self.input_size, hidden_size= self.hidden_size)
    self.softmax= nn.LogSoftmax(dim=1)
  
  
  def forward(self, input_seq, hidden_input):
    if (hidden_input.shape[1]!= self.hidden_size):
      hidden_input= self.comp(hidden_input)
    #print(input_seq.shape)
    #print(hidden_input.shape)
    output, hidden_update= self.rnn_layer(input_seq, hidden_input)
    output= self.softmax(self.output_layer(output))
    #print(output.shape)
    return output, hidden_update

def validate_step(model, rv_tokenizer, image_feat, max_len, vocab_size, device, start_token):
  transform= T.ToPILImage()
  img= transform(image_feat)
  img.show()
  #plt.imshow(image)
  flag= True
  sent= "sos "
  word_got_ind= 1
  input_ini= torch.zeros((1, max_len))

  input_ini[0, start_token]= 1.0
  input_ini= input_ini.to(device)
  image_feat= image_feat.squeeze(0)
  image_feat= image_feat.to(device)

  while(flag and word_got_ind<max_len):
    #print(input_ini)
    output, hidden= model(input_ini, image_feat)
    
    output_token= torch.argmax(output, dim= 1)
    input_ini[0, word_got_ind] = output_token.item()
    word_got_ind+=1
    #print(output_token.item())
    #print(input_ini)
    sent+= f"{rv_tokenizer[output_token.item()]} "
    
    print(sent)

    if(rv_tokenizer[output_token.item()] == 'eos'):
      flag= False

def train(train_dl, vocab_size, max_len, rv_tokenizer, start_token):
    model= RNN_Model_(1024, max_len, vocab_size)
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model= model.to(device)
    criterion= nn.NLLLoss()
    lr= 1e-4
    optimizer= torch.optim.SGD(model.parameters(), lr= lr)
    torch.autograd.set_detect_anomaly(True)
    #print(next(iter(train_dl)))
    for epoch in range(3):
        epoch_loss= 0
        num_updates= 0
        for ind, sample in enumerate(train_dl):
            
            input_rnn, output_rnn= sample
            image_feat = input_rnn[0]
            input_seq= input_rnn[1]
            if((ind+1)% 50 == 0):
              model.eval()
              validate_step(model, rv_tokenizer, image_feat, max_len, vocab_size, device, start_token)
              model.train()
              
            #hidden_input = image_feat
            #hidden_input= torch.Tensor(hidden_input).squeeze(0)
            print(ind)
            if((ind+1)%10 == 0):
                print(f"EPOCH : {epoch + 1} Image : {ind + 1} : Total Loss : {epoch_loss/((num_updates))}")

            for caption in range(5):
                loss = torch.Tensor([0]).to(device)
                loss= Variable(loss, requires_grad= True)
                
                hidden_input= image_feat
                hidden_input= torch.Tensor(hidden_input).squeeze(0)
                hidden_input= hidden_input.to(device)
                num_updates+= len(input_seq[caption])
                for input_ind in range(len(input_seq[caption])):          
                      act_output= output_rnn[caption][input_ind]
                      act_input= input_seq[caption][input_ind]
                      act_input= act_input.to(torch.float).to(device)
                      act_output= act_output.to(device) 
                      hidden_input= image_feat
                      hidden_input= torch.Tensor(hidden_input).squeeze(0).to(device)
                      output, hidden_input= model(act_input, hidden_input)
                    
                      act_output= torch.argmax(act_output, dim= 1)
                        
                      l= criterion(output, act_output)/len(input_seq[caption])
                      loss=loss+ l
                

                optimizer.zero_grad()
                
                loss.backward(retain_graph= False)
                optimizer.step()
                #for p in model.parameters():
                #    p.data.add_(p.grad.data, alpha=-lr)
                #epoch_loss= 0
                epoch_loss+= loss.item()
        
        print(f"Epoch : {epoch+ 1} ---- LOSS : {epoch_loss/(ind+1)}")

if __name__ == "__main__":
    
    
    train_dl, max_len, vocab_size, rv_tokenizer, start_token= utils.get_all_required_data()
    

    train(train_dl, vocab_size, max_len, rv_tokenizer, start_token)


