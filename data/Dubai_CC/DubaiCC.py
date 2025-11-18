import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from imageio import imread
from random import randint
from preprocess_data import encode  

class DubaiCCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, list_path, split,
                 token_folder=None, vocab_file=None,
                 max_length=80, allow_unk=0, max_iters=None):
        self.mean = [156.0681, 115.0891, 92.0745]
        self.std = [76.5139, 52.0616, 44.4209]
        self.split = split
        self.max_length = max_length

        assert split in {'train', 'val', 'test'}
        self.img_ids = [i.strip() for i in open(os.path.join(list_path, split + '.txt'))]

        if vocab_file:
            with open(os.path.join(list_path, vocab_file + '.json'), 'r') as f:
                self.word_vocab = json.load(f)
            self.allow_unk = allow_unk

        if max_iters:
            n = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n
            self.img_ids = self.img_ids[:max_iters]

        self.files = []
        if split == 'train':
            for entry in self.img_ids:
              
                file_part, token_id = entry.rsplit('-', 1)     # "11x37.tif", "2"
                stem, ext = os.path.splitext(file_part)       # "11x37", ".tif"

                imgA = os.path.join(data_folder, '500_2000',        file_part)
                imgB = os.path.join(data_folder, '500_2010',        file_part)
                fineA = os.path.join(data_folder, '500_2000fine',   stem + '.npy')
                fineB = os.path.join(data_folder, '500_2010fine',   stem + '.npy')
                semA  = os.path.join(data_folder, '500_2000semantic', stem + '.npy')
                semB  = os.path.join(data_folder, '500_2010semantic', stem + '.npy')
                token_file = os.path.join(token_folder, stem + '.txt') if token_folder else None

                self.files.append({
                    "imgA": imgA, "imgB": imgB,
                    "fineA": fineA, "fineB": fineB,
                    "semA": semA,   "semB": semB,
                    "token": token_file,
                    "token_id": token_id,
                    "name": file_part
                })
        else:
            for file_part in self.img_ids:
                stem, ext = os.path.splitext(file_part)
                imgA = os.path.join(data_folder, '500_2000',        file_part)
                imgB = os.path.join(data_folder, '500_2010',        file_part)
                fineA = os.path.join(data_folder, '500_2000fine',   stem + '.npy')
                fineB = os.path.join(data_folder, '500_2010fine',   stem + '.npy')
                semA  = os.path.join(data_folder, '500_2000semantic', stem + '.npy')
                semB  = os.path.join(data_folder, '500_2010semantic', stem + '.npy')
                token_file = os.path.join(token_folder, stem + '.txt') if token_folder else None

                self.files.append({
                    "imgA": imgA, "imgB": imgB,
                    "fineA": fineA, "fineB": fineB,
                    "semA": semA,   "semB": semB,
                    "token": token_file,
                    "token_id": None,
                    "name": file_part
                })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        name = datafiles["name"]

   
        imgA = np.asarray(imread(datafiles["imgA"]), np.float32)
        imgB = np.asarray(imread(datafiles["imgB"]), np.float32)
        imgA = np.moveaxis(imgA, -1, 0)
        imgB = np.moveaxis(imgB, -1, 0)
        for i in range(3):
            imgA[i] = (imgA[i] - self.mean[i]) / self.std[i]
            imgB[i] = (imgB[i] - self.mean[i]) / self.std[i]

        
        def load_feature(feature_file):
          
            if os.path.exists(feature_file):
                return torch.tensor(np.load(feature_file), dtype=torch.float32)
            else:
                return torch.zeros(1)

        fine_featureA     = load_feature(datafiles["fineA"])
        fine_featureB     = load_feature(datafiles["fineB"])
        semantic_featureA = load_feature(datafiles["semA"])
        semantic_featureB = load_feature(datafiles["semB"])

     
        token_all = np.zeros(1, dtype=int)
        token = np.zeros(1, dtype=int)
        token_len = np.zeros(1, dtype=int)
        token_all_len = np.zeros(1, dtype=int)
        if datafiles["token"]:
            with open(datafiles["token"], 'r') as f:
                captions = json.load(f)
            token_all = np.zeros((len(captions), self.max_length), int)
            token_all_len = np.zeros((len(captions), 1), int)
            for j, cap in enumerate(captions):
                enc = encode(cap, self.word_vocab, allow_unk=self.allow_unk==1)
                token_all[j, :len(enc)] = enc
                token_all_len[j] = len(enc)
            if datafiles["token_id"] is not None:
                idx = int(datafiles["token_id"])
                token = token_all[idx]
                token_len = token_all_len[idx].item()
            else:
                j = randint(0, len(captions)-1)
                token = token_all[j]
                token_len = token_all_len[j].item()

        return (
            imgA.copy(), imgB.copy(),
            token_all.copy(), token_all_len.copy(),
            token.copy(), np.array(token_len),
            name,
            fine_featureA, fine_featureB,
            semantic_featureA, semantic_featureB
        )


if __name__ == '__main__':
    ds = DubaiCCDataset(
        data_folder='Dubai_CC',
        list_path='./data/Dubai_CC',
        split='train',
        token_folder='./data/Dubai_CC/tokens'
    )
    loader = DataLoader(ds, batch_size=4, shuffle=True, pin_memory=True)
    for batch in loader:
        imgA, imgB, token_all, token_all_len, token, token_len, names, fA, fB, sA, sB = batch
        print('imgA', imgA.shape,
              'fA', fA.shape,
              'sA', sA.shape)
        break
