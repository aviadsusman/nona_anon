import torch.utils.data as td
from datasets import Dataset
import pandas as pd
import torch
from transformers import AutoTokenizer
import torchvision.transforms as transforms
from PIL import Image

class RSNADataset(td.Dataset):
    def __init__(self, indices, scaler=None):
        super(RSNADataset, self).__init__()
        self.indices = indices
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        # self.scaler = scaler
        self.scaler = [1, 228]

        self.features = pd.read_csv('data/rsna/all_features.csv')
        self.features = self.features[self.features['id'].isin(self.indices)].reset_index()
    
    def __len__(self):
        return len(self.indices)

    def _scale_label(self, label):
        if self.scaler == None:
            labels = self.features[self.features['id'].isin(self.indices)]['boneage']
            min_label, max_label = labels.min(), labels.max()
            self.scaler = [min_label, max_label]
        else:
            min_label, max_label = self.scaler[0], self.scaler[1]
        
        return (label - min_label) / (max_label - min_label)
    
    def __getitem__(self, idx):
        img_path = self.features.loc[idx, 'path']
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)
        label = self.features.loc[idx, 'boneage']
        
        return image, self._scale_label(label)

class AdressoDataset:
    def __init__(self, ids=None):
        self.ids = ids

        full_df = pd.read_parquet('data/adresso/x_y.parquet')

        # test and rest are labeled differently
        id_char = 'd' if self.ids is None else 'o'
        df = full_df[full_df['id'].str[4] == id_char]

        if self.ids is not None: # not test
            df = df[df['id'].isin(self.ids)]

        self.df = df
        self.dataset = Dataset.from_pandas(self.df)

        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        extra_cols = ["id", "dx", "mmse", "mmse binned", "path", "text", "__index_level_0__"]
        self.dataset = self.dataset.map(self.process_example, remove_columns=extra_cols)

    def len(self):
        return len(self.df)

    def process_example(self, example):
        tokenized = self.tokenizer(example["text"], padding="max_length", truncation=True)

        label = example['mmse']

        tokenized["labels"] = label
        return tokenized

    def get_dataset(self):
        return self.dataset

class DDSMDataset(td.Dataset):
    def __init__(self, indices):
        super(DDSMDataset, self).__init__()
        self.indices = indices
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
            ])

        self.data = pd.read_excel('data/ddsm/Data_final.xlsx')
        self.data = self.data.iloc[self.indices]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        path = self.data.iloc[idx]['fullPath']
        image = Image.open(path)
        if self.transform:
            image = self.transform(image)
        label = self.data.iloc[idx]['Age norm']
        
        return image, label

class UdacityDataset(td.Dataset):
    def __init__(self, indices):
        super(UdacityDataset, self).__init__()
        self.indices = indices
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
            ])

        self.data = pd.read_csv('data/udacity/data_df.csv')
        self.data = self.data.iloc[self.indices]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        path = self.data.iloc[idx]['image_path']
        image = Image.open(path)
        if self.transform:
            image = self.transform(image)
        label = self.data.iloc[idx]['steering_angle_norm']
        
        return image, label

class PitchforkDataset:
    def __init__(self, ids=None):
        self.ids = ids

        df = pd.read_csv('data/pitchfork/reviews.csv')
        df = df[df.index.isin(ids)]

        self.df = df
        self.dataset = Dataset.from_pandas(self.df)

        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        self.dataset = self.dataset.map(self.process_example)
    
    def len(self):
        return len(self.df)

    def process_example(self, example):
        tokenized = self.tokenizer(example["text"], padding="max_length", truncation=True)

        tokenized["labels"] = example["label"]
        return tokenized

    def get_dataset(self):
        return self.dataset

def collate_img(batch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x, y = zip(*batch)
    x = torch.stack(x).to(device).to(torch.float32)
    y = torch.tensor(y, dtype=torch.float32, device=device)
        
    return x, y

def collate_txt(batch):
    input_ids = torch.tensor([item["input_ids"] for item in batch])
    attention_mask = torch.tensor([item["attention_mask"] for item in batch])
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.float)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def collate_audio(batch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x, y = zip(*batch)
    x = torch.stack(x).to(device).to(torch.float32)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    return {'input_values': x}, y

def collate(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs, labels = zip(*batch)
    inputs = torch.stack(inputs).to(device)
    labels = torch.tensor(labels).to(device)
    
    return inputs, labels


def data_loaders(train, val, test, dataset, label='mmse', batch_size=32):
    # Necessary to shuffle training data so samples can see new
    # neighbors between batches.
    if dataset == 'rsna':
        train_dataset = RSNADataset(train)
        train_loader = td.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_img)
        
        val_dataset = RSNADataset(val)
        val_loader = td.DataLoader(val_dataset, batch_size=128, collate_fn=collate_img)
        
        test_dataset = RSNADataset(test)
        test_loader = td.DataLoader(test_dataset, batch_size=128, collate_fn=collate_img)


    if dataset == 'adresso':
        train_dataset = AdressoDataset(ids=train)
        train_loader = td.DataLoader(train_dataset.get_dataset(), batch_size=4, shuffle=True, collate_fn=collate_txt)

        val_dataset = AdressoDataset(ids=val)
        val_loader = td.DataLoader(val_dataset.get_dataset(), batch_size=val_dataset.len(), collate_fn=collate_txt)

        test_dataset = AdressoDataset()
        test_loader = td.DataLoader(test_dataset.get_dataset(), batch_size=test_dataset.len(), collate_fn=collate_txt)
    
    elif dataset == 'coughvid':
        train_loader = td.DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_audio)
        val_loader = td.DataLoader(val, batch_size=128, collate_fn=collate_audio)
        test_loader = td.DataLoader(test, batch_size=128, collate_fn=collate_audio)
    
    elif dataset == 'nosemic':
        train_loader = td.DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate)
        val_loader = td.DataLoader(val, batch_size=128, collate_fn=collate)
        test_loader = td.DataLoader(test, batch_size=128, collate_fn=collate)
    
    elif dataset == 'ddsm':
        train_dataset = DDSMDataset(indices=train)
        train_loader = td.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_img)

        val_dataset = DDSMDataset(indices=val)
        val_loader = td.DataLoader(val_dataset, batch_size=128, collate_fn=collate_img)

        test_dataset = DDSMDataset(indices=test)
        test_loader = td.DataLoader(test_dataset, batch_size=128, collate_fn=collate_img)
    
    elif dataset == 'udacity':
        train_dataset = UdacityDataset(indices=train)
        train_loader = td.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_img)

        val_dataset = UdacityDataset(indices=val)
        val_loader = td.DataLoader(val_dataset, batch_size=128, collate_fn=collate_img)

        test_dataset = UdacityDataset(indices=test)
        test_loader = td.DataLoader(test_dataset, batch_size=128, collate_fn=collate_img)
    
    elif dataset == 'pitchfork':
        train_dataset = PitchforkDataset(ids=train)
        train_loader = td.DataLoader(train_dataset.get_dataset(), batch_size=batch_size, shuffle=True, collate_fn=collate_txt)

        val_dataset = PitchforkDataset(ids=val)
        val_loader = td.DataLoader(val_dataset.get_dataset(), batch_size=128, collate_fn=collate_txt)

        test_dataset = PitchforkDataset(ids=test)
        test_loader = td.DataLoader(test_dataset.get_dataset(), batch_size=128, collate_fn=collate_txt)
    
    return {'train': train_loader, 'val': val_loader, 'test': test_loader}