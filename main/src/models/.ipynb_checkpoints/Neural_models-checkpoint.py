import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import contextlib
import io
import os

from engression import engression
from src.utils import (
    EarlyStopping,
    train,
    train_no_early_stopping,
    train_trans,
    train_trans_no_early_stopping,
)

from rtdl_revisiting_models import MLP, ResNet, FTTransformer

# --- HELPER CLASSES ---

class FastTensorDataLoader:
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        assert all(t.device == tensors[0].device for t in tensors)
        self.tensors = tensors
        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len, device=self.tensors[0].device)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i : self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return (self.dataset_len + self.batch_size - 1) // self.batch_size

class _TorchBase:
    def __init__(self, batch_size: int = 32, classification: bool = False, learning_rate: float = 1e-3, weight_decay: float = 0.0, n_epochs: int = 100, patience: int = None, checkpoint_path: str = None, seed: int = None, device=None, num_classes=None):
        self.batch_size      = batch_size
        self.classification = classification
        self.learning_rate   = learning_rate
        self.weight_decay   = weight_decay
        self.n_epochs       = n_epochs
        self.patience       = patience
        self.checkpoint_path = checkpoint_path
        self.seed           = seed
        
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        self.n_classes = num_classes

    def _prepare(self, X, y=None):
        """Moves data to GPU and sanitizes it."""
        Xa = X.values if hasattr(X, "values") else np.asarray(X)
        if Xa.ndim == 1: Xa = Xa.reshape(-1, 1)
        
        Xt = torch.tensor(Xa, dtype=torch.float32, device=self.device)
        # Sanitize Inputs (Clamp 20 sigma)
        Xt = torch.nan_to_num(Xt, nan=0.0, posinf=20.0, neginf=-20.0)
        Xt = torch.clamp(Xt, min=-20.0, max=20.0)
        
        if y is None: return Xt
            
        ya = y.values if hasattr(y, "values") else np.asarray(y)
        if self.classification:
            yt = torch.tensor(ya.reshape(-1), dtype=torch.long, device=self.device)
            
            if self.n_classes is None: 
                self.n_classes = int(torch.unique(yt).numel())
            
            # Ensure labels are within [0, n_classes-1]
            if yt.max() >= self.n_classes or yt.min() < 0:
                 yt = torch.clamp(yt, min=0, max=self.n_classes - 1)
        else:
            yt = torch.tensor(ya.reshape(-1, 1), dtype=torch.float32, device=self.device)
            yt = torch.nan_to_num(yt, nan=0.0)
            
        return Xt, yt

    def _loader(self, Xt, yt, shuffle: bool):
        return FastTensorDataLoader(Xt, yt, batch_size=self.batch_size, shuffle=shuffle)

    def prepare_data(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
        X_tr_t, y_tr_t = self._prepare(X_train, y_train)
        if self.classification and hasattr(self, 'd_out') and self.d_out == 1:
            y_tr_t = y_tr_t.float().view(-1, 1)

        train_loader = self._loader(X_tr_t, y_tr_t, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            X_va_t, y_va_t = self._prepare(X_val, y_val)
            if self.classification and hasattr(self, 'd_out') and self.d_out == 1:
                y_va_t = y_va_t.float().view(-1, 1)
            val_loader = self._loader(X_va_t, y_va_t, shuffle=False)
            
        test_loader = None
        if X_test is not None and y_test is not None:
            X_te_t, y_te_t = self._prepare(X_test, y_test)
            if self.classification and hasattr(self, 'd_out') and self.d_out == 1:
                y_te_t = y_te_t.float().view(-1, 1)
            test_loader = self._loader(X_te_t, y_te_t, shuffle=False)

        self.d_in = X_tr_t.size(1)
        return train_loader, val_loader, test_loader

# ----- MLP Regressor -----
class MLPRegressor(_TorchBase):
    def __init__(self, n_blocks=2, d_block=128, dropout=0.5, learning_rate=1e-3, weight_decay=1e-5, batch_size=32, patience=10, checkpoint_path="checkpoint_mlp.pt", seed=None, n_epochs=100, device=None):
        super().__init__(batch_size=batch_size, classification=False, learning_rate=learning_rate, weight_decay=weight_decay, n_epochs=n_epochs, patience=patience, checkpoint_path=checkpoint_path, seed=seed, device=device)
        self.n_blocks = n_blocks; self.d_block = d_block; self.dropout = dropout; self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if self.seed is not None:
            torch.manual_seed(self.seed); torch.cuda.manual_seed_all(self.seed); np.random.seed(self.seed)

        train_loader, val_loader, _ = self.prepare_data(X_train, y_train, X_val, y_val)
        self.model = MLP(d_in=self.d_in, d_out=1, n_blocks=self.n_blocks, d_block=self.d_block, dropout=self.dropout).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        with contextlib.redirect_stdout(io.StringIO()):
            if val_loader is not None and self.patience and self.patience > 0:
                actual_epochs_run = train(self.model, criterion, optimizer, self.n_epochs, train_loader, val_loader, EarlyStopping(self.patience, self.checkpoint_path), self.checkpoint_path)
            else:
                actual_epochs_run = train_no_early_stopping(self.model, criterion, optimizer, self.n_epochs, train_loader)
        return actual_epochs_run

    def predict(self, X):
        Xt = self._prepare(X) 
        loader = FastTensorDataLoader(Xt, torch.zeros((Xt.size(0), 1), device=self.device), batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                preds.append(self.model(xb).reshape(-1))
        return torch.cat(preds)
    
    def predict_with_uncertainty(self, X, train_loader, sample_size: int = None):
        mu = self.predict(X) 
        self.model.eval()
        all_residuals = []
        with torch.no_grad():
            for xb, yb in train_loader:
                all_residuals.append(yb.view(-1) - self.model(xb).view(-1))
        sigma_t = torch.std(torch.cat(all_residuals), correction=1)
        if sample_size:
            samples_t = torch.normal(mean=mu.unsqueeze(1).expand(-1, sample_size), std=sigma_t)
            return mu, sigma_t, samples_t
        return mu, sigma_t


# ----- MLP Classifier -----
class MLPClassifier(_TorchBase):
    def __init__(self, num_classes: int, n_blocks=2, d_block=128, dropout=0.5, batch_size=32, learning_rate=1e-3, weight_decay=0.0, n_epochs=100, patience=10, checkpoint_path="checkpoint_mlp.pt", seed=None, device=None):
        super().__init__(batch_size=batch_size, classification=True, learning_rate=learning_rate, weight_decay=weight_decay, n_epochs=n_epochs, patience=patience, checkpoint_path=checkpoint_path, seed=seed, device=device, num_classes=num_classes)
        self.n_blocks = n_blocks; self.d_block = d_block; self.dropout = dropout; self.num_classes = num_classes; self.d_out = num_classes; self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if self.seed is not None:
            torch.manual_seed(self.seed); torch.cuda.manual_seed_all(self.seed); np.random.seed(self.seed)
        
        train_loader, val_loader, _ = self.prepare_data(X_train, y_train, X_val, y_val)
        self.model = MLP(d_in=self.d_in, d_out=self.d_out, n_blocks=self.n_blocks, d_block=self.d_block, dropout=self.dropout).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss() if self.d_out == 1 else nn.CrossEntropyLoss()
        
        with contextlib.redirect_stdout(io.StringIO()):
            if val_loader is not None and self.patience is not None and self.patience > 0:
                actual_epochs_run = train(self.model, criterion, optimizer, self.n_epochs, train_loader, val_loader, EarlyStopping(self.patience, self.checkpoint_path), self.checkpoint_path)
            else:
                actual_epochs_run = train_no_early_stopping(self.model, criterion, optimizer, self.n_epochs, train_loader)
        return actual_epochs_run

    def predict_proba(self, X):
        Xt = self._prepare(X)
        loader = FastTensorDataLoader(Xt, torch.zeros(len(Xt), device=self.device), batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        probs = []
        with torch.no_grad():
            for xb, _ in loader:
                logits = self.model(xb)
                probs.append(torch.sigmoid(logits.view(-1)) if self.d_out == 1 else torch.softmax(logits, dim=1))
        return torch.cat(probs, dim=0)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).int() if self.d_out == 1 else torch.argmax(probabilities, dim=1)


# ----- Engression Models -----
class EngressionRegressor(_TorchBase):
    def __init__(self, learning_rate=1e-4, num_epochs=500, num_layer=3, hidden_dim=128, resblock=False, batch_size=32, seed=None, device=None, standardize=False):
        super().__init__(batch_size=batch_size, classification=False, learning_rate=learning_rate, device=device)
        self.num_epochs = num_epochs; self.num_layer = num_layer; self.hidden_dim = hidden_dim; self.resblock = resblock; self.seed = seed
        self.model = None

    def fit(self, X_train, y_train):
        if self.seed is not None:
            torch.manual_seed(self.seed); torch.cuda.manual_seed_all(self.seed); np.random.seed(self.seed)
        Xt, yt = self._prepare(X_train, y_train)
        
        with contextlib.redirect_stdout(io.StringIO()):
            self.model = engression(Xt, yt, lr=self.learning_rate, num_epochs=self.num_epochs, num_layer=self.num_layer, hidden_dim=self.hidden_dim, noise_dim=self.hidden_dim, batch_size=self.batch_size, resblock=self.resblock, device=str(self.device), standardize=False)
        return self

    def predict(self, X):
        Xt = self._prepare(X)
        return self.model.predict(Xt, target="mean").view(-1)
    
    def predict_samples(self, X, sample_size: int = 100):
        Xt = self._prepare(X)
        samples = self.model.sample(Xt, sample_size=sample_size, expand_dim=False)
        return samples.squeeze(1) if samples.shape[1] == 1 else samples

    def predict_with_uncertainty(self, X, train_loader, sample_size: int = None):
        mu = self.predict(X) 
        residuals = []
        for Xb, yb in train_loader:
            residuals.append(yb.view(-1) - self.model.predict(Xb, target="mean").view(-1))
        sigma_t = torch.std(torch.cat(residuals), correction=1)
        if sample_size: 
            return mu, sigma_t, self.predict_samples(X, sample_size)
        return mu, sigma_t


class EngressionClassifier(_TorchBase):
    def __init__(self, learning_rate=1e-4, num_epochs=500, num_layer=3, hidden_dim=128, resblock=False, batch_size=32, seed=None, device=None, num_classes=None):
        super().__init__(batch_size=batch_size, classification=True, learning_rate=learning_rate, device=device, num_classes=num_classes)
        self.num_epochs = num_epochs; self.num_layer = num_layer; self.hidden_dim = hidden_dim; self.resblock = resblock; self.seed = seed
        self.model = None; self.ovr_models = None; self.classes_t = None

    def fit(self, X_train, y_train):
        # Prepare data (will use self.n_classes correctly now)
        Xt, yt_full = self._prepare(X_train, y_train)
        
        if self.n_classes:
            self.classes_t = torch.arange(self.n_classes, device=self.device)
        else:
            y_np = (y_train.values if hasattr(y_train, "values") else np.asarray(y_train)).reshape(-1)
            self.classes_ = np.array(sorted(np.unique(y_np)))
            self.n_classes = int(len(self.classes_))
            self.classes_t = torch.tensor(self.classes_, device=self.device)

        if self.seed is not None:
            torch.manual_seed(self.seed); torch.cuda.manual_seed_all(self.seed); np.random.seed(self.seed)

        with contextlib.redirect_stdout(io.StringIO()):
            if self.n_classes <= 2:
                # Binary Case: Fast
                y01 = yt_full.float() if yt_full.max() <= 1 else (yt_full == yt_full.max()).float()
                yt = y01.view(-1, 1)
                self.model = engression(Xt, yt, classification=True, lr=self.learning_rate, num_epochs=self.num_epochs, num_layer=self.num_layer, hidden_dim=self.hidden_dim, noise_dim=self.hidden_dim, batch_size=self.batch_size, resblock=self.resblock, device=str(self.device))
            else:
                # Multiclass Case: One-Vs-Rest
                # OPTIMIZATION: Create one-hot targets ON GPU to avoid 24x CPU transfers
                self.ovr_models = []
                
                # y_onehot shape: (N, n_classes)
                y_onehot = torch.nn.functional.one_hot(yt_full, num_classes=self.n_classes).float()
                
                for cls_idx in range(self.n_classes):
                    # Slice column for this class -> (N, 1)
                    yt = y_onehot[:, cls_idx].view(-1, 1)
                    
                    self.ovr_models.append(engression(Xt, yt, classification=True, lr=self.learning_rate, num_epochs=self.num_epochs, num_layer=self.num_layer, hidden_dim=self.hidden_dim, noise_dim=self.hidden_dim, batch_size=self.batch_size, resblock=self.resblock, device=str(self.device)))
        return self

    def predict_proba(self, X):
        Xt = self._prepare(X)
        if self.n_classes <= 2:
            raw = self.model.predict(Xt, target="mean").view(-1)
            if (raw.min() < 0) or (raw.max() > 1): raw = torch.sigmoid(raw)
            return raw.detach().view(-1)
        probs = []
        with torch.no_grad():
            for m in self.ovr_models:
                r = m.predict(Xt, target="mean").view(-1)
                if (r.min() < 0) or (r.max() > 1): r = torch.sigmoid(r)
                probs.append(r)
        P = torch.stack(probs, dim=1); P = P / (P.sum(dim=1, keepdim=True) + 1e-8)
        return P.detach()

    def predict(self, X):
        probs = self.predict_proba(X)
        if self.n_classes <= 2:
            return (probs >= 0.5).int()
        else:
            return self.classes_t[torch.argmax(probs, dim=1)]


# ----- ResNet Models -----
class ResNetRegressor(_TorchBase):
    def __init__(self, n_blocks=2, d_block=128, d_hidden_multiplier=1.0, dropout1=0.5, dropout2=0.5, learning_rate=1e-3, weight_decay=0.0, batch_size=32, patience=10, checkpoint_path="checkpoint_resnet.pt", seed=None, n_epochs=100, device=None):
        super().__init__(batch_size=batch_size, classification=False, learning_rate=learning_rate, weight_decay=weight_decay, n_epochs=n_epochs, patience=patience, checkpoint_path=checkpoint_path, seed=seed, device=device)
        self.n_blocks = n_blocks; self.d_block = d_block; self.d_hidden_multiplier = d_hidden_multiplier; self.dropout1 = dropout1; self.dropout2 = dropout2; self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if self.seed is not None:
            torch.manual_seed(self.seed); torch.cuda.manual_seed_all(self.seed); np.random.seed(self.seed)
        train_loader, val_loader, _ = self.prepare_data(X_train, y_train, X_val, y_val)
        self.d_in = train_loader.tensors[0].size(1)
        self.model = ResNet(d_in=self.d_in, d_out=1, n_blocks=self.n_blocks, d_block=self.d_block, d_hidden=None, d_hidden_multiplier=self.d_hidden_multiplier, dropout1=self.dropout1, dropout2=self.dropout2).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        with contextlib.redirect_stdout(io.StringIO()):
            epochs_run = train(self.model, criterion, optimizer, self.n_epochs, train_loader, val_loader, EarlyStopping(self.patience, self.checkpoint_path), self.checkpoint_path) if val_loader and self.patience else train_no_early_stopping(self.model, criterion, optimizer, self.n_epochs, train_loader)
        return epochs_run

    def predict(self, X):
        Xt = self._prepare(X)
        loader = FastTensorDataLoader(Xt, torch.zeros((Xt.size(0), 1), device=self.device), batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loader: preds.append(self.model(xb).reshape(-1))
        return torch.cat(preds)
        
    def predict_with_uncertainty(self, X, train_loader, sample_size: int = None):
        mu = self.predict(X)
        self.model.eval()
        residuals = []
        with torch.no_grad():
            for xb, yb in train_loader: residuals.append(yb.view(-1) - self.model(xb).view(-1))
        return mu, torch.std(torch.cat(residuals), correction=1)

class ResNetClassifier(_TorchBase):
    def __init__(self, num_classes: int, n_blocks=2, d_block=128, d_hidden_multiplier=1.0, dropout1=0.5, dropout2=0.5, learning_rate=1e-3, weight_decay=0.0, batch_size=32, patience=10, checkpoint_path="checkpoint_resnet.pt", seed=None, n_epochs=100, device=None):
        super().__init__(batch_size=batch_size, classification=True, learning_rate=learning_rate, weight_decay=weight_decay, n_epochs=n_epochs, patience=patience, checkpoint_path=checkpoint_path, seed=seed, device=device, num_classes=num_classes)
        self.n_blocks = n_blocks; self.d_block = d_block; self.d_hidden_multiplier = d_hidden_multiplier; self.dropout1 = dropout1; self.dropout2 = dropout2; self.num_classes = num_classes; self.d_out = self.num_classes; self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if self.seed is not None:
            torch.manual_seed(self.seed); torch.cuda.manual_seed_all(self.seed); np.random.seed(self.seed)
        train_loader, val_loader, _ = self.prepare_data(X_train, y_train, X_val, y_val)
        self.d_in = train_loader.tensors[0].size(1)
        self.model = ResNet(d_in=self.d_in, d_out=self.d_out, n_blocks=self.n_blocks, d_block=self.d_block, d_hidden=None, d_hidden_multiplier=self.d_hidden_multiplier, dropout1=self.dropout1, dropout2=self.dropout2).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss() if self.d_out == 1 else nn.CrossEntropyLoss()

        with contextlib.redirect_stdout(io.StringIO()):
            epochs_run = train(self.model, criterion, optimizer, self.n_epochs, train_loader, val_loader, EarlyStopping(self.patience, self.checkpoint_path), self.checkpoint_path) if val_loader and self.patience else train_no_early_stopping(self.model, criterion, optimizer, self.n_epochs, train_loader)
        return epochs_run

    def predict_proba(self, X):
        Xt = self._prepare(X)
        loader = FastTensorDataLoader(Xt, torch.zeros((Xt.size(0), 1), device=self.device), batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        probs = []
        with torch.no_grad():
            for xb, _ in loader:
                logits = self.model(xb)
                probs.append(torch.sigmoid(logits) if self.d_out == 1 else torch.softmax(logits, dim=1))
        return torch.cat(probs, dim=0).reshape(-1) if self.d_out == 1 else torch.cat(probs, dim=0)

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= 0.5).int().flatten() if self.d_out == 1 else torch.argmax(probs, dim=1)


# ----- FTTransformer Models -----
class FTTrans_Regressor(_TorchBase):
    def __init__(self, n_blocks=2, d_block_multiplier=8, attention_n_heads=8, attention_dropout=0.5, ffn_d_hidden_multiplier=2.0, ffn_dropout=0.5, residual_dropout=0.5, learning_rate=1e-3, weight_decay=0.0, batch_size=32, patience=10, checkpoint_path="checkpoint_fttrans.pt", seed=None, n_epochs=100, device=None):
        if batch_size < 1024: 
            batch_size = 4096 
        super().__init__(batch_size=batch_size, classification=False, learning_rate=learning_rate, weight_decay=weight_decay, n_epochs=n_epochs, patience=patience, checkpoint_path=checkpoint_path, seed=seed, device=device)
        self.n_blocks = n_blocks; self.d_block_multiplier = d_block_multiplier; self.attention_n_heads = int(attention_n_heads); self.attention_dropout = attention_dropout; self.ffn_d_hidden_multiplier = ffn_d_hidden_multiplier; self.ffn_dropout = ffn_dropout; self.residual_dropout = residual_dropout

    def fit(self, X_train, y_train, X_val = None, y_val = None):
        if self.seed is not None:
            torch.manual_seed(self.seed); torch.cuda.manual_seed_all(self.seed); np.random.seed(self.seed)
        train_loader, val_loader, _ = self.prepare_data(X_train, y_train, X_val, y_val)
        self.d_in = train_loader.tensors[0].size(1)
        
        base_d_block = int(self.d_block_multiplier * self.attention_n_heads)
        if base_d_block % self.attention_n_heads != 0:
            base_d_block = (base_d_block // self.attention_n_heads) * self.attention_n_heads
            
        ffn_hidden = max(2, int(base_d_block * self.ffn_d_hidden_multiplier))

        self.model = FTTransformer(
            n_cont_features=self.d_in, cat_cardinalities=[], d_out=1, n_blocks=self.n_blocks, 
            d_block=base_d_block, attention_n_heads=self.attention_n_heads, attention_dropout=self.attention_dropout, 
            ffn_d_hidden=ffn_hidden, ffn_d_hidden_multiplier=None, ffn_dropout=self.ffn_dropout, 
            residual_dropout=self.residual_dropout
        ).to(self.device)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        with contextlib.redirect_stdout(io.StringIO()):
            epochs_run = train_trans(self.model, criterion, optimizer, self.n_epochs, train_loader, val_loader, EarlyStopping(self.patience, self.checkpoint_path), self.checkpoint_path) if val_loader and self.patience else train_trans_no_early_stopping(self.model, criterion, optimizer, self.n_epochs, train_loader)
        return epochs_run
    
    def predict(self, X):
        Xt = self._prepare(X)
        loader = FastTensorDataLoader(Xt, torch.zeros((Xt.size(0), 1), device=self.device), batch_size=self.batch_size * 2, shuffle=False)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loader: preds.append(self.model(xb, None).reshape(-1))
        return torch.cat(preds)
    
    def predict_with_uncertainty(self, X, train_loader, sample_size: int = None):
        mu = self.predict(X); self.model.eval()
        residuals = []
        with torch.no_grad():
            for xb, yb in train_loader: residuals.append(yb.view(-1) - self.model(xb, None).view(-1))
        return mu, torch.std(torch.cat(residuals), correction=1)


class FTTrans_Classifier(_TorchBase):
    def __init__(self, num_classes: int, n_blocks=2, d_block_multiplier=8, attention_n_heads=8, attention_dropout=0.5, ffn_d_hidden_multiplier=0.5, ffn_dropout=0.5, residual_dropout=0.5, learning_rate=1e-3, weight_decay=0.0, batch_size=32, patience=10, checkpoint_path="checkpoint_fttrans.pt", seed=None, n_epochs=100, device=None):
        if batch_size < 1024: 
            batch_size = 4096 
        super().__init__(batch_size=batch_size, classification=True, learning_rate=learning_rate, weight_decay=weight_decay, n_epochs=n_epochs, patience=patience, checkpoint_path=checkpoint_path, seed=seed, device=device, num_classes=num_classes)
        self.n_blocks = n_blocks; self.d_block_multiplier = d_block_multiplier; self.attention_n_heads = int(attention_n_heads); self.attention_dropout = attention_dropout; self.ffn_d_hidden_multiplier = ffn_d_hidden_multiplier; self.ffn_dropout = ffn_dropout; self.residual_dropout = residual_dropout; self.num_classes = num_classes; self.d_out = self.num_classes

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if self.seed is not None:
            torch.manual_seed(self.seed); torch.cuda.manual_seed_all(self.seed); np.random.seed(self.seed)
        train_loader, val_loader, _ = self.prepare_data(X_train, y_train, X_val, y_val)
        self.d_in = train_loader.tensors[0].size(1)
        
        base_d_block = int(self.d_block_multiplier * self.attention_n_heads)
        if base_d_block % self.attention_n_heads != 0:
            base_d_block = (base_d_block // self.attention_n_heads) * self.attention_n_heads
            
        ffn_hidden = max(2, int(base_d_block * self.ffn_d_hidden_multiplier))
        
        self.model = FTTransformer(
            n_cont_features=self.d_in, cat_cardinalities=[], d_out=self.d_out, n_blocks=self.n_blocks, 
            d_block=base_d_block, attention_n_heads=self.attention_n_heads, attention_dropout=self.attention_dropout, 
            ffn_d_hidden=ffn_hidden, ffn_d_hidden_multiplier=None, ffn_dropout=self.ffn_dropout, 
            residual_dropout=self.residual_dropout
        ).to(self.device)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss() if self.d_out == 1 else nn.CrossEntropyLoss()

        with contextlib.redirect_stdout(io.StringIO()):
            epochs_run = train_trans(self.model, criterion, optimizer, self.n_epochs, train_loader, val_loader, EarlyStopping(self.patience, self.checkpoint_path), self.checkpoint_path) if val_loader and self.patience else train_trans_no_early_stopping(self.model, criterion, optimizer, self.n_epochs, train_loader)
        return epochs_run
    
    def predict_proba(self, X):
        Xt = self._prepare(X)
        loader = FastTensorDataLoader(Xt, torch.zeros((Xt.size(0), 1), device=self.device), batch_size=self.batch_size * 2, shuffle=False)
        self.model.eval()
        probs = []
        with torch.no_grad():
            for xb, _ in loader:
                logits = self.model(xb, None) 
                probs.append(torch.sigmoid(logits.view(-1)) if self.d_out == 1 else torch.softmax(logits, dim=1))
        return torch.cat(probs, dim=0).reshape(-1) if self.d_out == 1 else torch.cat(probs, dim=0)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).int().flatten() if self.d_out == 1 else torch.argmax(self.predict_proba(X), axis=1)