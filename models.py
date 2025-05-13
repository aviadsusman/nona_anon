import torch
import torch.nn as nn
from torch.nn.functional import softmax
import similarity as s

class NONA(nn.Module):
    '''
    Nearness of Neighbors Attention 
    A differentiable regressor inspired by attention and KNN.
    Given prediction data X and labeled neighbor data (X_N,y_N)
    In the notation of attention, Q = Fe(X), K = Fe(X_N) and V = y_N where Fe is an upstream feature extractor.
    In the notation of KNN, k = |X_N|, metric = -l2 distance or dot product, weights = softmax.
    '''
    def __init__(self, similarity='l2', softstep=None, batch_norm=None, dtype=torch.float64):
        super(NONA, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        
        self.similarity = similarity
        
        if softstep: self.softstep = softstep.to(self.device).to(self.dtype)
        else: self.softstep = softstep

        self.batch_norm = batch_norm # Num input features. Used for standalone NONA.
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(self.batch_norm, dtype=self.dtype, device=self.device)
    
    def softmax_predict(self, sim, y_n, train):
        '''
        Make predictions using sim matrix and neighbor labels.
        Correct for self-similarity during training.
        '''
        if train:
            inf_id = torch.diag(torch.full((len(sim),), torch.inf)).to(self.device)
            sim -= inf_id

        sim_scores = softmax(sim, dim=1)

        return sim_scores @ y_n

    def forward(self, x, x_n, y_n):
        if self.batch_norm:
            x = self.bn(x)
            x_n = self.bn(x_n)

        # Create similarity matrix between embeddings of X and embeddings of X_n
        if self.softstep:
            sim = self.softstep(x, x_n, similarity=self.similarity)
        else:
            sim = s.sim_matrix(x, x_n, similarity=self.similarity)

        train = torch.equal(x, x_n)
        output = self.softmax_predict(sim, y_n, train)
        return torch.clip(output, 0,1)

class NN(nn.Module):
    '''
    Attach intermediate MLP to either a NONA or dense prediction layer.
    '''
    def __init__(self, input_size, hl_sizes=list(), predictor='nona', similarity='l2', softstep=None, dtype=torch.float64):
        super(NN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype

        self.input_size = input_size
        self.hl_sizes = hl_sizes
        self.predictor = predictor # for benchmarking
        self.similarity = similarity
        self.softstep = softstep
        
        layer_dims = [self.input_size] + self.hl_sizes

        if hl_sizes != list():
            self.fcn = nn.ModuleList(nn.Linear(layer_dims[i], layer_dims[i+1], dtype=self.dtype, device=self.device) for i in range(len(layer_dims)-1))
        
            self.activation = nn.Tanh() # Tanh allows for negative feature covariance between samples

            self.norms = nn.ModuleList(nn.BatchNorm1d(i, dtype=self.dtype, device=self.device) for i in layer_dims[:-1])

        if self.predictor=='nona':
            self.output_layer = NONA(similarity=self.similarity, softstep=self.softstep, dtype=self.dtype)
        
        elif self.predictor=='dense':
            self.output_layer = nn.Linear(layer_dims[-1], 1, dtype=self.dtype, device=self.device)

    def forward(self, x, x_n, y_n, embeddings=False):
        if self.hl_sizes != list():
            for layer, norm in zip(self.fcn, self.norms):
                x = self.activation(layer(norm(x)))

                if self.predictor=='nona' or embeddings==True:
                    x_n = self.activation(layer(norm(x_n)))

        output = self.output_layer(x, x_n, y_n).squeeze()
        if embeddings:
            output = [output, x, x_n]
        
        return output

class Finetuner(nn.Module):
    def __init__(self, feature_extractor, hl_sizes=list(), predictor='nona', similarity=None, softstep=None, dtype=torch.float64):
        '''
        Attach a feature extractor to a NONA neural network.
        '''
        super(Finetuner, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        
        self.feature_extractor = feature_extractor.to(self.device)
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        self.fe_type = self._detect_fe_type()
        
        
        self.input_size = self._strip_and_get_in_features()
        self.hl_sizes = hl_sizes
        
        self.predictor = predictor
        self.similarity = similarity
        self.softstep = softstep

        self.inter = NN(
            input_size=self.input_size, 
            hl_sizes=self.hl_sizes,
            predictor=self.predictor, 
            similarity=self.similarity, 
            softstep=self.softstep,
            dtype=self.dtype)

    def _detect_fe_type(self):
        fe = self.feature_extractor
        if hasattr(fe, 'config') and hasattr(fe.config, 'model_type'):
            return fe.config.model_type.lower()
        elif isinstance(fe, nn.Module):
            if hasattr(fe, 'encoder') and callable(getattr(fe, 'forward', None)):
                return 'whisper'
            return 'cnn'
        else:
            raise ValueError("Unsupported model type.")
    
    def _strip_and_get_in_features(self):
        fe = self.feature_extractor

        # DistilBERT, HuBERT
        if hasattr(fe, 'config') and hasattr(fe.config, 'hidden_size'):
            return fe.config.hidden_size
        
        if self.fe_type == 'modernbert':
            return fe.config.hidden_size

        if self.fe_type == 'whisper':
            with torch.no_grad():
                dummy = torch.zeros(1, 80, 3000).to(next(fe.parameters()).device)
                out = fe(dummy)
                return out.shape[-1]

        # ResNet
        if hasattr(fe, 'fc') and isinstance(fe.fc, nn.Linear):
            in_features = fe.fc.in_features
            fe.fc = nn.Identity()
            return in_features

        # EfficientNet
        if hasattr(fe, 'classifier') and isinstance(fe.classifier, nn.Sequential):
            for i, layer in reversed(list(enumerate(fe.classifier))):
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features
                    fe.classifier[i] = nn.Identity()
                    return in_features

        # Swin
        if hasattr(fe, 'head') and isinstance(fe.head, nn.Linear):
            in_features = fe.head.in_features
            fe.head = nn.Identity()
            return in_features

        # Fallback: last Linear layer in model
        for module in reversed(list(fe.modules())):
            if isinstance(module, nn.Linear):
                return module.in_features

    def extract_features(self, inp):
        if self.fe_type in ['bert', 'distilbert']:
            out = self.feature_extractor(**inp)
            return out.last_hidden_state[:, 0, :]
        elif self.fe_type in ['hubert', 'wav2vec2']:
            out = self.feature_extractor(**inp)
            return out.last_hidden_state[:, 0, :]
        elif self.fe_type == 'cnn':
            out = self.feature_extractor(inp)
            if out.ndim == 4:
                out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
                out = out.view(out.size(0), -1)
            return out
        elif self.fe_type == 'whisper':
            return self.feature_extractor(inp).mean(dim=1)
        else:
            raise ValueError(f"Unsupported model type: {self.fe_type}")

    def forward(self, x, x_n, y_n, embeddings=False):
        x = self.extract_features(x)
        if self.predictor == 'nona' or embeddings:
            x_n = self.extract_features(x_n)

        return self.inter(x, x_n, y_n, embeddings)