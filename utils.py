import torch
from torcheval.metrics.functional import mean_squared_error as mse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from transformers import AutoModel, HubertModel
import whisper
import torchvision
from torchvision.models import resnet18, swin_t, efficientnet_b0
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def fe_for_data(dataset):
    if dataset == 'rsna':
        return resnet18(weights='DEFAULT')

    elif dataset == 'adresso':
        return AutoModel.from_pretrained('distilbert-base-uncased')

    elif dataset == 'coughvid':
        return HubertModel.from_pretrained('facebook/hubert-base-ls960')

    elif dataset == 'nosemic':
        return whisper.load_model('tiny').encoder

    elif dataset == 'ddsm':
        return swin_t(weights='IMAGENET1K_V1')

    elif dataset == 'udacity':
        return efficientnet_b0(pretrained=True)

    elif dataset == 'pitchfork':
        return AutoModel.from_pretrained('bert-base-uncased')

def folds(dataset, seed, label='mmse'):
    if dataset == 'rsna':
        data_df = pd.read_csv('data/rsna/data.csv')
        fold_dict = {}
        ids = data_df['id'].values
        binned_labels = data_df['boneage binned'].values

        tv_ids, fold_dict['test'], tv_labels, _ = train_test_split(ids, binned_labels, test_size=0.2, stratify=binned_labels, random_state=seed)
        
        fold_dict['train'], fold_dict['val'], _, _ = train_test_split(tv_ids, tv_labels, stratify=tv_labels, test_size=0.15, random_state=seed)

        return fold_dict

    elif dataset == 'adresso':
        data_df = pd.read_parquet('data/adresso/x_y.parquet')
        fold_dict = {}
        ids = data_df['id'].values
        if label == 'mmse':
            splitting_labels = data_df['mmse binned'].values
        elif label == 'dx':
            splitting_labels = data_df['dx'].values

        fold_dict['train'], fold_dict['val'] = train_test_split(ids, test_size=0.15, stratify=splitting_labels, random_state=seed)

        fold_dict['test'] = None

        return fold_dict
    
    elif dataset == 'coughvid':
        X = torch.load('data/coughvid/audio.pt')
        y = torch.load('data/coughvid/ages.pt')
        binned_labels = torch.load('data/coughvid/binned_ages.pt')
        ids = np.arange(len(y)).astype(int)
        tv_ids, test_ids, tv_labels, _ = train_test_split(ids, binned_labels, test_size=0.2, stratify=binned_labels, random_state=seed)

        train_ids, val_ids, _, _ = train_test_split(tv_ids, tv_labels, test_size=0.15, stratify=tv_labels, random_state=seed)

        fold_dict = {'train': TensorDataset(X[train_ids],y[train_ids]),
                     'val': TensorDataset(X[val_ids], y[val_ids]),
                     'test': TensorDataset(X[test_ids],y[test_ids])
                     }

        return fold_dict
    
    elif dataset == 'nosemic':
        X = torch.load('data/nosemic/processed_nosemic_audio.pt')
        y = torch.load('data/nosemic/processed_nosemic_labels.pt')
        binned_labels = torch.load('data/nosemic/binned_labels.pt')
        ids = np.arange(len(y)).astype(int)
        tv_ids, test_ids, tv_labels, _ = train_test_split(ids, binned_labels, test_size=0.2, stratify=binned_labels, random_state=seed)

        train_ids, val_ids, _, _ = train_test_split(tv_ids, tv_labels, test_size=0.15, stratify=tv_labels, random_state=seed)

        fold_dict = {'train': TensorDataset(X[train_ids],y[train_ids]),
                     'val': TensorDataset(X[val_ids], y[val_ids]),
                     'test': TensorDataset(X[test_ids],y[test_ids])
                     }
        return fold_dict
    
    elif dataset == 'ddsm':
        fold_dict = {}
        data = pd.read_excel('data/ddsm/Data_final.xlsx')
        labels = data['Age'] # Enough of each age to split on
        ids = data['ID']

        tv_ids, test_ids, tv_labels, _ = train_test_split(ids, labels, test_size=0.2, stratify=labels, random_state=seed)
        train_ids, val_ids, _, _ = train_test_split(tv_ids, tv_labels, test_size=0.15, stratify=tv_labels, random_state=seed)

        fold_dict['train'] = data[data['ID'].isin(train_ids)].index.tolist()
        fold_dict['val'] = data[data['ID'].isin(val_ids)].index.tolist()
        fold_dict['test'] = data[data['ID'].isin(test_ids)].index.tolist()
        
        return fold_dict

    elif dataset == 'udacity':
        fold_dict = {}
        data = pd.read_csv('data/udacity/data_df.csv')
        data = data.sample(n=15_000)
        labels = data['steering_angle_norm'] # Enough of each age to split on
        ids = list(data.index)
        
        tv_ids, fold_dict['test'], tv_labels, _ = train_test_split(ids, labels, test_size=0.2, random_state=seed)
        fold_dict['train'], fold_dict['val'], _, _ = train_test_split(tv_ids, tv_labels, test_size=0.15, random_state=seed)
        
        return fold_dict
    
    elif dataset == 'pitchfork':
        fold_dict = {}
        data = pd.read_csv('data/pitchfork/reviews.csv')
        data = data.sample(n=1500)
        labels = data['label'] # Enough of each age to split on
        ids = list(data.index)
        
        tv_ids, fold_dict['test'], tv_labels, _ = train_test_split(ids, labels, test_size=0.2, random_state=seed)
        fold_dict['train'], fold_dict['val'], _, _ = train_test_split(tv_ids, tv_labels, test_size=0.15, random_state=seed)
        
        return fold_dict


def tune_knn(X_train, X_test, y_train, y_test, task):

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.cpu().detach())

    X_test = scaler.transform(X_test.cpu().detach())

    scorer = make_scorer(mean_squared_error, greater_is_better=False) 
 
    knn = KNeighborsRegressor()

    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        scoring=scorer,
        cv=4,
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train.cpu().detach().squeeze())

    best_knn = grid_search.best_estimator_

    y_hat_knn = best_knn.predict(X_test)

    return torch.tensor(y_hat_knn, dtype=y_test.dtype).to(y_test.device)

def sliced(data):
    if isinstance(data, torch.Tensor):
        return data[:2]
    elif isinstance(data, dict):
        return {k:v[:2] for k,v in data.items()}