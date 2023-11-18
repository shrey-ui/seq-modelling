import numpy as np 
import os 
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.transforms import transforms
from transformers import BertTokenizer
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.utils import to_categorical


def get_caption(path_of_caption_file):
    captions = pd.read_csv(path_of_caption_file, delimiter=',', header= 0)
    dict_captions= {key: [] for key in captions['image']}
    for ind, row in captions.iterrows():
        dict_captions[row['image']].append(row['caption'])
    return dict_captions, captions

def file2img(img_path):
    img= Image.open(f'./data/Flickr8k/Images/{img_path}')
    trans= transforms.Compose([transforms.ToTensor()])
    img_tensor= trans(img)
    return img_tensor
class ImageCaptionDataset(Dataset):
    def __init__(self, caption_file_path, tokenizer, max_length, vocab_size):
        self.file_names= os.listdir('./data/Flickr8k/Images/')
        self.caption_file_path= caption_file_path
        self.captions_dict, self.caption_df= get_caption(caption_file_path)
        self.tokenizer= tokenizer
        self.max_length= max_length
        self.vocab_size= vocab_size

    def __len__(self):
        return len(self.captions_dict)

    def seq2seq(self, sequence):
        inp_combs= []
        out_corr= []
        for i in range(1, len(sequence)):
            inp_seq = sequence[:i]
            out_seq= sequence[i]
    
            inp_seq= pad_sequences([inp_seq], maxlen= self.max_length)[0]
            out_seq= to_categorical([out_seq], num_classes= self.vocab_size)[0]
            inp_combs.append(inp_seq)
            out_corr.append(out_seq)
        return np.array(inp_combs), np.array(out_corr)   

    def __getitem__(self, idx):
        img= self.file_names[idx]
        captions = self.captions_dict[img]
        image_tens= file2img(img)
        inp1 = []
        out= []
        for capt_examp in captions:
            print(capt_examp)
            tokens= self.tokenizer.texts_to_sequences([capt_examp])[0]
            inp_seq, out_seq= ImageCaptionDataset.seq2seq(self, tokens)
            inp1.append(inp_seq)
            out.append(out_seq)
        
        return [image_tens, inp1], out

def create_tokenizer(caption_corpus):
    tokenizer= Tokenizer()
    tokenizer.fit_on_texts(caption_corpus)
    return tokenizer

if __name__ == '__main__':
    dict_of_capts, captions_df= get_caption('./data/Flickr8k/captions.txt')
    capt_list= []
    for file in dict_of_capts:
        [capt_list.append(cap) for cap in dict_of_capts[file]]
    tokenizer= create_tokenizer(capt_list)
    
    vocab_size= len(tokenizer.word_index) + 1
    max_len= max(len(d.split()) for d in capt_list)

    CaptionDataset= ImageCaptionDataset('./data/Flickr8k/captions.txt', 
                                        tokenizer, max_len,
                                        vocab_size)

    train_dl= DataLoader(CaptionDataset, batch_size= 1, shuffle= True)
    print(list(train_dl)[0]) 

