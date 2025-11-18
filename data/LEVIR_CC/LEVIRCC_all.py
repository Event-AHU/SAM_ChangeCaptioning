import torch
from torch.utils.data import Dataset
from preprocess_data import encode
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
#import cv2 as cv
from imageio import imread
from random import *

class LEVIRCCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, list_path, split, token_folder=None, vocab_file=None, max_length=40, allow_unk=0, max_iters=None):
        """
        :param data_folder: folder where image files are stored
        :param list_path: folder where the file name-lists of Train/val/test.txt sets are stored
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param token_folder: folder where token files are stored
        :param vocab_file: the name of vocab file
        :param max_length: the maximum length of each caption sentence
        :param max_iters: the maximum iteration when loading the data
        :param allow_unk: whether to allow the tokens have unknow word or not
        """
        self.mean = [100.6790,  99.5023,  84.9932]
        self.std = [50.9820, 48.4838, 44.7057]
        self.list_path = list_path
        self.split = split
        self.max_length = max_length
        
        self.data_folder = data_folder
        

        if split == 'train':
            self.fine_feat_dirA = os.path.join(self.data_folder, 'fine_features_trainA1')
            self.fine_feat_dirB = os.path.join(self.data_folder, 'fine_features_trainB1')
            self.semantic_feat_dirA=os.path.join(self.data_folder, 'semantic_trainA')
            self.semantic_feat_dirB=os.path.join(self.data_folder, 'semantic_trainB')
        elif split == 'val':
            self.fine_feat_dirA = os.path.join(self.data_folder, 'fine_features_valA2')
            self.fine_feat_dirB = os.path.join(self.data_folder, 'fine_features_valB2')
            self.semantic_feat_dirA=os.path.join(self.data_folder, 'semantic_valA')
            self.semantic_feat_dirB=os.path.join(self.data_folder, 'semantic_valB')
        else:

            self.fine_feat_dirA = os.path.join(self.data_folder, 'fine_features_testA3')
            self.fine_feat_dirB = os.path.join(self.data_folder, 'fine_features_testB3')
            self.semantic_feat_dirA=os.path.join(self.data_folder, 'semantic_testA')
            self.semantic_feat_dirB=os.path.join(self.data_folder, 'semantic_testB')


        assert self.split in {'train', 'val', 'test'}
        self.img_ids = [i_id.strip() for i_id in open(os.path.abspath(os.path.join(list_path, split + '.txt')))]
        if vocab_file is not None:
            with open(os.path.abspath(os.path.join(list_path, vocab_file + '.json')), 'r') as f:
                self.word_vocab = json.load(f)
            self.allow_unk = allow_unk
        if not max_iters == None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters - n_repeat * len(self.img_ids)]
        self.files = []
        if split == 'train':
            for name in self.img_ids:
                img_fileA = os.path.abspath(os.path.join(data_folder, split, 'A', name.split('-')[0]))
                img_fileB = img_fileA.replace('A', 'B')
                token_id = name.split('-')[-1]
                if token_folder is not None:
                    token_file = os.path.abspath(os.path.join(token_folder, name.split('.')[0] + '.txt'))
                else:
                    token_file = None
                self.files.append({
                    "imgA": img_fileA,
                    "imgB": img_fileB,
                    "token": token_file,
                    "token_id": token_id,
                    "name": name.split('-')[0]
                })
        elif split == 'val':
            for name in self.img_ids:
                img_fileA = os.path.abspath(os.path.join(data_folder, split, 'A', name))
                img_fileB = img_fileA.replace('A', 'B')
                token_id = None
                if token_folder is not None:
                    token_file = os.path.abspath(os.path.join(token_folder, name.split('.')[0] + '.txt'))
                else:
                    token_file = None
                self.files.append({
                    "imgA": img_fileA,
                    "imgB": img_fileB,
                    "token": token_file,
                    "token_id": token_id,
                    "name": name
                })
        elif split == 'test':
            for name in self.img_ids:
                img_fileA = os.path.abspath(os.path.join(data_folder, split, 'A', name))
                img_fileB = img_fileA.replace('A', 'B')
                token_id = None
                if token_folder is not None:
                    token_file = os.path.abspath(os.path.join(token_folder, name.split('.')[0] + '.txt'))
                else:
                    token_file = None
                self.files.append({
                    "imgA": img_fileA,
                    "imgB": img_fileB,
                    "token": token_file,
                    "token_id": token_id,
                    "name": name
                })
    


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        name = datafiles["name"]
        imgA = imread(datafiles["imgA"])
        imgB = imread(datafiles["imgB"])
        imgA = np.asarray(imgA, np.float32)
        imgB = np.asarray(imgB, np.float32)   
        imgA = np.moveaxis(imgA, -1, 0)     
        imgB = np.moveaxis(imgB, -1, 0)

        for i in range(len(self.mean)):
            imgA[i,:,:] -= self.mean[i]
            imgA[i,:,:] /= self.std[i]
            imgB[i,:,:] -= self.mean[i]
            imgB[i,:,:] /= self.std[i]     
        
                
                 
        if datafiles["token"] is not None:
            caption = open(datafiles["token"])
            caption = caption.read()
            caption_list = json.loads(caption)



            token_all = np.zeros((len(caption_list),self.max_length),dtype=int)
            token_all_len = np.zeros((len(caption_list),1),dtype=int)
            for j, tokens in enumerate(caption_list):
                tokens_encode = encode(tokens, self.word_vocab,
                                    allow_unk=self.allow_unk == 1)
                token_all[j,:len(tokens_encode)] = tokens_encode
                token_all_len[j] = len(tokens_encode)
            if datafiles["token_id"] is not None:
                id = int(datafiles["token_id"])
                token = token_all[id]
                token_len = token_all_len[id].item()
            else:
                j = randint(0, len(caption_list) - 1)
                token = token_all[j]
                token_len = token_all_len[j].item()
        else:
            token_all = np.zeros(1,dtype=int)
            token = np.zeros(1,dtype=int)
            token_len = np.zeros(1,dtype=int)
            token_all_len = np.zeros(1,dtype=int)
            
        fine_featureA, fine_featureB = None, None
        if self.fine_feat_dirA is not None and self.fine_feat_dirB is not None:
            base_name, _ = os.path.splitext(name)
            featA_path = os.path.join(self.fine_feat_dirA, base_name + '.npy')
            featB_path = os.path.join(self.fine_feat_dirB, base_name + '.npy')
            if os.path.exists(featA_path):
                fine_featureA = np.load(featA_path)
            if os.path.exists(featB_path):
                fine_featureB = np.load(featB_path)
            if fine_featureA is not None:
                fine_featureA = torch.from_numpy(fine_featureA).float()
            if fine_featureB is not None:
                fine_featureB = torch.from_numpy(fine_featureB).float() 

        semantic_featureA, semantic_featureB = None, None
        if self.semantic_feat_dirA is not None and self.semantic_feat_dirB is not None:
            base_name, _ = os.path.splitext(name)
            sfeatA_path = os.path.join(self.semantic_feat_dirA, base_name + '.npy')
            sfeatB_path = os.path.join(self.semantic_feat_dirB, base_name + '.npy')
            if os.path.exists(sfeatA_path):
                semantic_featureA = np.load(sfeatA_path)
            if os.path.exists(sfeatB_path):
                semantic_featureB = np.load(sfeatB_path)
            if semantic_featureA is not None:
                semantic_featureA = torch.from_numpy(semantic_featureA).float()
            if semantic_featureB is not None:
                semantic_featureB = torch.from_numpy(semantic_featureB).float()            

        return imgA.copy(), imgB.copy(), token_all.copy(), token_all_len.copy(), token.copy(), np.array(token_len), name, fine_featureA.clone(), fine_featureB.clone(),semantic_featureA.clone(), semantic_featureB.clone()

if __name__ == '__main__':
    
    train_dataset = LEVIRCCDataset(data_folder='LEVIR_CC/images',list_path='./data/LEVIR_CC/', split= 'train', token_folder=None)
    train_loader = DataLoader(dataset=train_dataset,batch_size=1,shuffle=False,pin_memory=True)
    channels_sumA,channel_squared_sumA,channels_sumB,channel_squared_sumB = 0,0,0,0
    num_batches = len(train_loader)
    index = 0
    for dataA,dataB,_,_,_,_,_ in train_loader:
        index += 1
        if index%1000==0:
           print(index,num_batches)
        channels_sumA += torch.mean(dataA,dim=[0,2,3])   
        channel_squared_sumA += torch.mean(dataA**2,dim=[0,2,3])       
        channels_sumB += torch.mean(dataB,dim=[0,2,3])   
        channel_squared_sumB += torch.mean(dataB**2,dim=[0,2,3])
        channels_sum = channels_sumA + channels_sumB
        channel_squared_sum = channel_squared_sumA + channel_squared_sumB
    meanA = channels_sumA/num_batches
    stdA = (channel_squared_sumA/num_batches - meanA**2)**0.5
    meanB = channels_sumB/num_batches
    stdB = (channel_squared_sumB/num_batches - meanB**2)**0.5
    mean = (channels_sum)/(num_batches*2)
    std = ((channel_squared_sum) / (num_batches*2) - mean**2)**0.5   
    print(meanA,stdA,meanB,stdB,mean,std) 
