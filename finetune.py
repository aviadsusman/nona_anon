import torch
from torch.nn import MSELoss
from torcheval.metrics.functional import mean_squared_error as mse
from data.dataset_classes import data_loaders
from torch.optim import Adam
from models import Finetuner, NONA
import similarity as s
from utils import *
import argparse
import time
from copy import deepcopy
import pickle as pkl
from tqdm import tqdm
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Helper functions
def sliced(data):
    if isinstance(data, torch.Tensor):
        return data[:2]
    elif isinstance(data, dict):
        return {k:v[:2] for k,v in data.items()}

def x_y(batch):
    if isinstance(batch, dict):
        X = {key: val.to(device) for key, val in batch.items() if key!='labels'}
        y = batch['labels'].to(device)      
    else:
        X,y = batch

    return X, y

def z_y(loader, desc):
    z_fold = []
    y_fold = []
    for batch in tqdm(loader, desc=desc, file=sys.stdout):
        X, y = x_y(batch)
        y_fold.append(y)
        _, z, _ = model(X, sliced(X), sliced(y), embeddings=True)
        z_fold.append(z)
    return torch.cat(z_fold), torch.cat(y_fold)

# Main functions
def build_model(feature_extractor, predictor, softstep_params, d=50):
    if predictor == 'dense':
        similarity = None
    else:
        similarity = predictor
        predictor = 'nona'
    
    hls = [200, d]

    if softstep_params:
        softstep = s.SoftStep(**softstep_params)
    else:
        softstep = None

    model = Finetuner(feature_extractor=fe, 
                    hl_sizes=hls, 
                    predictor=predictor, 
                    similarity=similarity, 
                    softstep=softstep,
                    dtype=torch.float32
                    )
    return model

def train_eval(train, val, test, model):
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-5)

    # Early stopping params
    patience = 10
    start_after_epoch = 5
    count = 0
    best_val_score = float('-inf')
    
    # Train loop
    start = time.time()
    epoch = 1
    while count < patience: 
        model.train()
        train_loss = 0.0
        print('Epoch:', epoch)
        for batch in tqdm(train, desc="Train", file=sys.stdout):
            X,y = x_y(batch)
            outputs = model(X, X, y) 

            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train)
        report = f"Train Loss: {train_loss: .5f}"

        # Early stopping on val set
        if epoch > start_after_epoch:
            model.eval()
            y_val = []
            y_hat_val = []
            with torch.no_grad():
                if predictor=='dense':
                    for batch in tqdm(val, desc='Val', file=sys.stdout):
                        X,y = x_y(batch)
                        y_val.append(y)
                        y_hat = model(X, X, y)
                        y_hat_val.append(y_hat)
                else:
                    z_train, y_train = z_y(train, 'Train for val')
                    for batch in tqdm(val, desc='Val', file=sys.stdout):
                        X,y = x_y(batch)
                        y_val.append(y)
                        _, z, _ = model(X, sliced(X), sliced(y), embeddings=True)
                        y_hat = model.inter.output_layer(z, z_train, y_train)
                        y_hat_val.append(y_hat)
                    
            y_val = torch.cat(y_val)
            y_hat_val = torch.cat(y_hat_val)

            val_score = mse(y_hat_val, y_val).item()
            if val_score < best_val_score:
                best_val_score = val_score
                best_model_state = deepcopy(model.state_dict())
                count = 0
            else:
                count += 1
            report = report + f': Val Score: {abs(val_score): .5f}'
        
        print(report)
        if isinstance(mask, s.UniformSoftMask):
            mask_params.append(deepcopy(mask.params.data).detach())
        
        epoch += 1

    # Eval on test set and extract embeddings for KNN
    print("Evaluating on test") 
    model.load_state_dict(best_model_state)
    z_test = []
    y_test = []
    y_hat_test = []
    with torch.no_grad():
        z_train, y_train = z_y(train, 'Final train embeddings')
        z_val, y_val = z_y(val, 'Final val embeddings') 

        for batch in tqdm(test, desc='Test predictions', file=sys.stdout):
            X,y = x_y(batch)
            y_test.append(y)
            _, z, _ = model(X, sliced(X), sliced(y), embeddings=True)
            z_test.append(z)

            if isinstance(model.inter.output_layer, NONA):
                y_hat = model.inter.output_layer(z, z_train, y_train)
            else:
                y_hat = model(X,X,y)
            y_hat_test.append(y_hat)
        
    z_test = torch.cat(z_test)
    y_test = torch.cat(y_test)
    y_hat_test = torch.cat(y_hat_test)

    end = time.time()
    
    model_objs = {'z_train': z_train, 'y_train': y_train,
                  'z_val': z_val, 'y_val': y_val,
                  'z_test': z_test, 'y_test': y_test,
                  'y_hat_test': y_hat_test,
                  'score': mse(y_hat_test, y_test).item(),
                  'time': end-start}

    print(f"Test score: {model_objs['score']}")
    if mask:
        model_objs['mask weights'] = mask.state_dict() 
    if save_models:
        model_objs['model weights'] = model.state_dict()
    if isinstance(mask, s.UniformSoftMask):
        model_objs['mask params'] = mask_params

    print(f"Training and evaluating tuned knn on final embeddings.") 
    start = time.time()
    y_hat_knn = tune_knn(z_train, z_test, y_train, y_test, task)
    end = time.time()
    model_objs['knn score and time'] = [mse(y_hat_knn, y_test).item(), end-start]

    return model_objs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset and configs.')
    parser.add_argument('--dataset', type=str, default='adresso', help='Which dataset.')
    parser.add_argument('--predictor', type=str, default='l2', help='prediction head')
    parser.add_argument('--abt', type=str, default=None, help='How to generate softstep params')
    parser.add_argument('--step_fn', type=int, default=1, help='Which step fn for softstep.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--embed', type=int, default=50, help='Embedding space size')
    parser.add_argument('--seed', type=int, default=0, help='Random split seed.')
    parser.add_argument('--start_time', type=str, default=None, help='Batch job start time for saving results.')
    parser.add_argument('--savemodels', action='store_true', default=False, help='Whether to save final models')
    
    args = parser.parse_args()
    dataset = args.dataset
    predictor = args.predictor
    d = args.embed
    if args.step_fn:
        softstep_params = {'step_fn': args.step_fn, 'dims': args.abt}
    else:
        softstep_params = None
    batch_size = args.batch_size
    seed = args.seed
    start_time = args.start_time
    if start_time is None:
        start_time = time.strftime("%m%d%H%M")
    save_models = args.savemodels
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    fe = fe_for_data(dataset)

    folds_dict = folds(dataset=dataset, seed=seed)

    loaders = data_loaders(**folds_dict, 
                           dataset=dataset,
                           batch_size=batch_size)

    model = build_model(feature_extractor=fe, 
                        predictor=predictor,
                        softstep_params=softstep_params,
                        d=d)

    model_objs = train_eval(**loaders, model=model)
    
    results_dir = f'results/{dataset}/{start_time}/{predictor}'
    results_file = f'{mask}_{step_fn}_mask_embed_{d}_bs_{batch_size}_{seed}.pth'
    results_path = os.path.join(results_dir, results_file)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    torch.save(model_objs, results_path)