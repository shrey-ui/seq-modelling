import torch.nn as nn
import torch
import matplotlib.pyplot as plt

from init import ALL_LETTERS, N_LETTERS
from init import load_data, letter_to_tensor, line_to_tensor, random_training_example

def get_class(output):
    return all_categories[torch.argmax(output).item()]


class RNN_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_Model, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o= nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim =1)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), dim=1)
        hidden_update= self.i2h(combined)
        output= self.softmax(self.i2o(combined))
        return output, hidden_update
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)



category_lines, all_categories= load_data()
n_categories= len(all_categories)
hidden_size= 128
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rnn= RNN_Model(N_LETTERS, hidden_size, n_categories).to(device)

criterion= nn.NLLLoss()
learning_rate= 0.01
optimizer = torch.optim.SGD(rnn.parameters(),lr= learning_rate)

def train(word_tensor, category_tensor):
    hidden= rnn.init_hidden()
    for i in range(word_tensor.size()[0]):
        shifted= word_tensor[i].to(device)
        output, hidden = rnn(shifted, hidden)

    loss = criterion(output, category_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()


current_loss= 0
all_losses= []
plot_steps, print_steps = 1000, 5000
n_iters= 100000
for i in range(n_iters):
    category,lines,category_tensor, line_tensor= random_training_example(category_lines, all_categories)
    output, loss = train(line_tensor, category_tensor)
    current_loss+= loss

    if (i+1)%plot_steps == 0:
        all_losses.append(current_loss/plot_steps)
        current_loss=0 

    if (i+1)%print_steps == 0:
        guess = get_class(output)
        print(f"Actual : {category}, Guess : {get_class(output)}")
        





