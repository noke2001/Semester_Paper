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

class DeviceDataLoader:
    """Wraps a DataLoader to move data to a device on the fly."""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl: 
            yield [x.to(self.device, non_blocking=True) for x in b]

    def __len__(self):
        return len(self.dl)

class _TorchBase:
    def __init__(self, batch_size: int = 32, classification: bool = False, learning_rate: float = 1e-3, weight_decay: float = 0.0, n_epochs: int = 100, patience: int = None, checkpoint_path: str = None, seed: int = None):
        self.batch_size      = batch_size
        self.classification = classification
        self.learning_rate   = learning_rate
        self.weight_decay   = weight_decay
        self.n_epochs       = n_epochs
        self.patience       = patience
        self.checkpoint_path = checkpoint_path
        self.seed           = seed
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_classes      = None

    def _prepare(self, X, y=None):
        Xa = X.values if hasattr(X, "values") else np.asarray(X)
        if Xa.ndim == 1: Xa = Xa.reshape(-1, 1)
        Xt = torch.tensor(Xa, dtype=torch.float32)
        
        if y is None: return Xt
            
        ya = y.values if hasattr(y, "values") else np.asarray(y)
        if self.classification:
            yt = torch.tensor(ya.reshape(-1), dtype=torch.long)
            if self.n_classes is None: self.n_classes = int(torch.unique(yt).numel())
        else:
            yt = torch.tensor(ya.reshape(-1, 1), dtype=torch.float32)
        return Xt, yt

    def _loader(self, Xt, yt, shuffle: bool, drop_last: bool = False):
        ds = TensorDataset(Xt, yt)
        # num_workers=0 is safer for stability, pin_memory helps transfer speed
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, drop_last=drop_last, pin_memory=True, num_workers=0)
        return DeviceDataLoader(dl, self.device)

    def prepare_data(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
        X_tr_t, y_tr_t = self._prepare(X_train, y_train)
        if self.classification and hasattr(self, 'd_out') and self.d_out == 1:
            y_tr_t = y_tr_t.float().view(-1, 1)

        train_loader = self._loader(X_tr_t, y_tr_t, shuffle=True, drop_last=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            X_va_t, y_va_t = self._prepare(X_val, y_val)
            if self.classification and hasattr(self, 'd_out') and self.d_out == 1:
                y_va_t = y_va_t.float().view(-1, 1)
            val_loader = self._loader(X_va_t, y_va_t, shuffle=False, drop_last=False)
            
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
    def __init__(self, n_blocks=2, d_block=128, dropout=0.5, learning_rate=1e-3, weight_decay=1e-5, batch_size=32, patience=10, checkpoint_path="checkpoint_mlp.pt", seed=None, n_epochs=100):
        super().__init__(batch_size=batch_size, classification=False, learning_rate=learning_rate, weight_decay=weight_decay, n_epochs=n_epochs, patience=patience, checkpoint_path=checkpoint_path, seed=seed)
        self.n_blocks = n_blocks; self.d_block = d_block; self.dropout = dropout; self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if self.seed is not None:
            torch.manual_seed(self.seed); torch.cuda.manual_seed_all(self.seed); np.random.seed(self.seed)

        train_loader, val_loader, _ = self.prepare_data(X_train, y_train, X_val, y_val)
        self.model = MLP(d_in=self.d_in, d_out=1, n_blocks=self.n_blocks, d_block=self.d_block, dropout=self.dropout).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        # SUPPRESS OUTPUT HERE
        with contextlib.redirect_stdout(io.StringIO()):
            if val_loader is not None and self.patience and self.patience > 0:
                actual_epochs_run = train(self.model, criterion, optimizer, self.n_epochs, train_loader, val_loader, EarlyStopping(self.patience, self.checkpoint_path), self.checkpoint_path)
            else:
                actual_epochs_run = train_no_early_stopping(self.model, criterion, optimizer, self.n_epochs, train_loader)
        return actual_epochs_run

    def predict(self, X):
        Xt = self._prepare(X).to(self.device)
        dummy_y = torch.zeros((Xt.size(0), 1), dtype=torch.float32)
        loader = self._loader(Xt.cpu(), dummy_y, shuffle=False)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                preds.append(self.model(xb).reshape(-1))
        return torch.cat(preds).cpu().numpy()
    
    def predict_with_uncertainty(self, X, train_loader, sample_size: int = None):
        mu = self.predict(X) 
        self.model.eval()
        all_residuals = []
        with torch.no_grad():
            for xb, yb in train_loader:
                all_residuals.append(yb.view(-1) - self.model(xb).view(-1))
        sigma = torch.std(torch.cat(all_residuals), correction=1).item()
        if sample_size:
            samples = np.random.normal(loc=mu[:, None], scale=sigma, size=(len(mu), sample_size))
            return mu, sigma, samples
        return mu, sigma


# ----- MLP Classifier -----
class MLPClassifier(_TorchBase):
    def __init__(self, num_classes: int,  n_blocks=2, d_block=128, dropout=0.5, batch_size=32, learning_rate=1e-3, weight_decay=0.0, n_epochs=100, patience=10, checkpoint_path="checkpoint_mlp.pt", seed=None):
        super().__init__(batch_size=batch_size, classification=True, learning_rate=learning_rate, weight_decay=weight_decay, n_epochs=n_epochs, patience=patience, checkpoint_path=checkpoint_path, seed=seed)
        self.n_blocks = n_blocks; self.d_block = d_block; self.dropout = dropout; self.num_classes = num_classes; self.d_out = num_classes; self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if self.seed is not None:
            torch.manual_seed(self.seed); torch.cuda.manual_seed_all(self.seed); np.random.seed(self.seed)
        
        train_loader, val_loader, _ = self.prepare_data(X_train, y_train, X_val, y_val)
        self.model = MLP(d_in=self.d_in, d_out=self.d_out, n_blocks=self.n_blocks, d_block=self.d_block, dropout=self.dropout).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss() if self.d_out == 1 else nn.CrossEntropyLoss()
        
        # SUPPRESS OUTPUT HERE
        with contextlib.redirect_stdout(io.StringIO()):
            if val_loader is not None and self.patience is not None and self.patience > 0:
                actual_epochs_run = train(self.model, criterion, optimizer, self.n_epochs, train_loader, val_loader, EarlyStopping(self.patience, self.checkpoint_path), self.checkpoint_path)
            else:
                actual_epochs_run = train_no_early_stopping(self.model, criterion, optimizer, self.n_epochs, train_loader)
        return actual_epochs_run

    def predict_proba(self, X):
        Xt = self._prepare(X)
        loader = self._loader(Xt, torch.zeros(len(Xt)), shuffle=False)
        probs = []
        self.model.eval()
        with torch.no_grad():
            for Xb, _ in loader:
                logits = self.model(Xb)
                probs.append(torch.sigmoid(logits.view(-1)) if self.d_out == 1 else torch.softmax(logits, dim=1))
        return torch.cat(probs, dim=0).cpu().numpy()

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int) if self.d_out == 1 else np.argmax(probabilities, axis=1)


# ----- Engression Models -----
class EngressionRegressor(_TorchBase):
    def __init__(self, learning_rate=1e-4, num_epochs=500, num_layer=3, hidden_dim=128, resblock=False, batch_size=32, seed=None, device=None, standardize=False):
        super().__init__(batch_size=batch_size, classification=False, learning_rate=learning_rate)
        self.num_epochs = num_epochs; self.num_layer = num_layer; self.hidden_dim = hidden_dim; self.resblock = resblock; self.seed = seed
        self.device = torch.device(device) if device else self.device; self.model = None

    def fit(self, X_train, y_train):
        if self.seed is not None:
            torch.manual_seed(self.seed); torch.cuda.manual_seed_all(self.seed); np.random.seed(self.seed)
        Xt, yt = self._prepare(X_train, y_train)
        Xt, yt = Xt.to(self.device), yt.to(self.device)

        # SUPPRESS OUTPUT HERE
        with contextlib.redirect_stdout(io.StringIO()):
            self.model = engression(Xt, yt, lr=self.learning_rate, num_epochs=self.num_epochs, num_layer=self.num_layer, hidden_dim=self.hidden_dim, noise_dim=self.hidden_dim, batch_size=self.batch_size, resblock=self.resblock, device=str(self.device), standardize=False)
        return self

    def predict(self, X):
        Xt = self._prepare(X).to(self.device)
        return self.model.predict(Xt, target="mean").cpu().numpy().reshape(-1)
    
    def predict_samples(self, X, sample_size: int = 100):
        Xt = self._prepare(X).to(self.device)
        samples = self.model.sample(Xt, sample_size=sample_size, expand_dim=False)
        return samples.squeeze(1).cpu().numpy() if samples.shape[1] == 1 else samples.cpu().numpy()

    def predict_with_uncertainty(self, X, train_loader, sample_size: int = None):
        mu = self.predict(X)
        residuals = []
        for Xb, yb in train_loader:
            residuals.append(yb.view(-1) - self.model.predict(Xb, target="mean").view(-1))
        sigma = torch.std(torch.cat(residuals), correction=1).item()
        if sample_size: return mu, sigma, self.predict_samples(X, sample_size)
        return mu, sigma


class EngressionClassifier(_TorchBase):
    def __init__(self, learning_rate=1e-4, num_epochs=500, num_layer=3, hidden_dim=128, resblock=False, batch_size=32, seed=None, device=None):
        super().__init__(batch_size=batch_size, classification=True, learning_rate=learning_rate)
        self.num_epochs = num_epochs; self.num_layer = num_layer; self.hidden_dim = hidden_dim; self.resblock = resblock; self.seed = seed
        self.device = torch.device(device) if device else self.device; self.model = None; self.ovr_models = None          

    def fit(self, X_train, y_train):
        Xt, _ = self._prepare(X_train, y_train); Xt = Xt.to(self.device)
        y_np = (y_train.values if hasattr(y_train, "values") else np.asarray(y_train)).reshape(-1)
        self.classes_ = np.array(sorted(np.unique(y_np))); self.n_classes_ = int(len(self.classes_))
        if self.seed is not None:
            torch.manual_seed(self.seed); torch.cuda.manual_seed_all(self.seed); np.random.seed(self.seed)

        # SUPPRESS OUTPUT HERE
        with contextlib.redirect_stdout(io.StringIO()):
            if self.n_classes_ == 2:
                y01 = y_np.astype(np.float32) if set(np.unique(y_np)) == {0, 1} else (y_np == self.classes_[1]).astype(np.float32)
                yt = torch.tensor(y01, dtype=torch.float32, device=self.device).view(-1, 1)
                self.model = engression(Xt, yt, classification=True, lr=self.learning_rate, num_epochs=self.num_epochs, num_layer=self.num_layer, hidden_dim=self.hidden_dim, noise_dim=self.hidden_dim, batch_size=self.batch_size, resblock=self.resblock, device=str(self.device))
            else:
                self.ovr_models = []
                for cls in self.classes_:
                    yt = torch.tensor((y_np == cls).astype(np.float32), dtype=torch.float32, device=self.device).view(-1, 1)
                    self.ovr_models.append(engression(Xt, yt, classification=True, lr=self.learning_rate, num_epochs=self.num_epochs, num_layer=self.num_layer, hidden_dim=self.hidden_dim, noise_dim=self.hidden_dim, batch_size=self.batch_size, resblock=self.resblock, device=str(self.device)))
        return self

    def predict_proba(self, X):
        Xt = self._prepare(X).to(self.device)
        if self.n_classes_ == 2:
            raw = self.model.predict(Xt, target="mean").view(-1)
            if (raw.min() < 0) or (raw.max() > 1): raw = torch.sigmoid(raw)
            return raw.detach().cpu().numpy().reshape(-1)
        probs = []
        with torch.no_grad():
            for m in self.ovr_models:
                r = m.predict(Xt, target="mean").view(-1)
                if (r.min() < 0) or (r.max() > 1): r = torch.sigmoid(r)
                probs.append(r)
        P = torch.stack(probs, dim=1); P = P / (P.sum(dim=1, keepdim=True) + 1e-8)
        return P.detach().cpu().numpy()

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int) if self.n_classes_ == 2 else self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ----- ResNet Models -----
class ResNetRegressor(_TorchBase):
    def __init__(self, n_blocks=2, d_block=128, d_hidden_multiplier=1.0, dropout1=0.5, dropout2=0.5, learning_rate=1e-3, weight_decay=0.0, batch_size=32, patience=10, checkpoint_path="checkpoint_resnet.pt", seed=None, n_epochs=100):
        super().__init__(batch_size=batch_size, classification=False, learning_rate=learning_rate, weight_decay=weight_decay, n_epochs=n_epochs, patience=patience, checkpoint_path=checkpoint_path, seed=seed)
        self.n_blocks = n_blocks; self.d_block = d_block; self.d_hidden_multiplier = d_hidden_multiplier; self.dropout1 = dropout1; self.dropout2 = dropout2; self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if self.seed is not None:
            torch.manual_seed(self.seed); torch.cuda.manual_seed_all(self.seed); np.random.seed(self.seed)
        train_loader, val_loader, _ = self.prepare_data(X_train, y_train, X_val, y_val)
        self.d_in = train_loader.dl.dataset.tensors[0].size(1)
        self.model = ResNet(d_in=self.d_in, d_out=1, n_blocks=self.n_blocks, d_block=self.d_block, d_hidden=None, d_hidden_multiplier=self.d_hidden_multiplier, dropout1=self.dropout1, dropout2=self.dropout2).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        # SUPPRESS OUTPUT HERE
        with contextlib.redirect_stdout(io.StringIO()):
            epochs_run = train(self.model, criterion, optimizer, self.n_epochs, train_loader, val_loader, EarlyStopping(self.patience, self.checkpoint_path), self.checkpoint_path) if val_loader and self.patience else train_no_early_stopping(self.model, criterion, optimizer, self.n_epochs, train_loader)
        return epochs_run

    def predict(self, X):
        Xt = self._prepare(X)
        loader = self._loader(Xt, torch.zeros((Xt.size(0), 1)), shuffle=False)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loader: preds.append(self.model(xb).reshape(-1))
        return torch.cat(preds).cpu().numpy()
        
    def predict_with_uncertainty(self, X, train_loader, sample_size: int = None):
        mu = self.predict(X)
        self.model.eval()
        residuals = []
        with torch.no_grad():
            for xb, yb in train_loader: residuals.append(yb.view(-1) - self.model(xb).view(-1))
        return mu, torch.std(torch.cat(residuals), correction=1).item()

class ResNetClassifier(_TorchBase):
    def __init__(self, num_classes: int, n_blocks=2, d_block=128, d_hidden_multiplier=1.0, dropout1=0.5, dropout2=0.5, learning_rate=1e-3, weight_decay=0.0, batch_size=32, patience=10, checkpoint_path="checkpoint_resnet.pt", seed=None, n_epochs=100):
        super().__init__(batch_size=batch_size, classification=True, learning_rate=learning_rate, weight_decay=weight_decay, n_epochs=n_epochs, patience=patience, checkpoint_path=checkpoint_path, seed=seed)
        self.n_blocks = n_blocks; self.d_block = d_block; self.d_hidden_multiplier = d_hidden_multiplier; self.dropout1 = dropout1; self.dropout2 = dropout2; self.num_classes = num_classes; self.d_out = self.num_classes; self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if self.seed is not None:
            torch.manual_seed(self.seed); torch.cuda.manual_seed_all(self.seed); np.random.seed(self.seed)
        train_loader, val_loader, _ = self.prepare_data(X_train, y_train, X_val, y_val)
        self.d_in = train_loader.dl.dataset.tensors[0].size(1)
        self.model = ResNet(d_in=self.d_in, d_out=self.d_out, n_blocks=self.n_blocks, d_block=self.d_block, d_hidden=None, d_hidden_multiplier=self.d_hidden_multiplier, dropout1=self.dropout1, dropout2=self.dropout2).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss() if self.d_out == 1 else nn.CrossEntropyLoss()

        # SUPPRESS OUTPUT HERE
        with contextlib.redirect_stdout(io.StringIO()):
            epochs_run = train(self.model, criterion, optimizer, self.n_epochs, train_loader, val_loader, EarlyStopping(self.patience, self.checkpoint_path), self.checkpoint_path) if val_loader and self.patience else train_no_early_stopping(self.model, criterion, optimizer, self.n_epochs, train_loader)
        return epochs_run

    def predict_proba(self, X):
        Xt = self._prepare(X)
        loader = self._loader(Xt, torch.zeros((Xt.size(0), 1)), shuffle=False)
        self.model.eval()
        probs = []
        with torch.no_grad():
            for xb, _ in loader:
                logits = self.model(xb)
                probs.append(torch.sigmoid(logits) if self.d_out == 1 else torch.softmax(logits, dim=1))
        return torch.cat(probs, dim=0).cpu().numpy().reshape(-1) if self.d_out == 1 else torch.cat(probs, dim=0).cpu().numpy()

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int).flatten() if self.d_out == 1 else np.argmax(self.predict_proba(X), axis=1)


# ----- FTTransformer Models -----
class FTTrans_Regressor(_TorchBase):
    def __init__(self, n_blocks=2, d_block_multiplier=8, attention_n_heads=8, attention_dropout=0.5, ffn_d_hidden_multiplier=2.0, ffn_dropout=0.5, residual_dropout=0.5, learning_rate=1e-3, weight_decay=0.0, batch_size=32, patience=10, checkpoint_path="checkpoint_fttrans.pt", seed=None, n_epochs=100):
        super().__init__(batch_size=batch_size, classification=False, learning_rate=learning_rate, weight_decay=weight_decay, n_epochs=n_epochs, patience=patience, checkpoint_path=checkpoint_path, seed=seed)
        self.n_blocks = n_blocks; self.d_block_multiplier = d_block_multiplier; self.attention_n_heads = int(attention_n_heads); self.attention_dropout = attention_dropout; self.ffn_d_hidden_multiplier = ffn_d_hidden_multiplier; self.ffn_dropout = ffn_dropout; self.residual_dropout = residual_dropout

    def fit(self, X_train, y_train, X_val = None, y_val = None):
        if self.seed is not None:
            torch.manual_seed(self.seed); torch.cuda.manual_seed_all(self.seed); np.random.seed(self.seed)
        train_loader, val_loader, _ = self.prepare_data(X_train, y_train, X_val, y_val)
        self.d_in = train_loader.dl.dataset.tensors[0].size(1)
        d_block = int(self.d_block_multiplier * self.attention_n_heads)
        ffn_hidden = max(2, int(d_block * self.ffn_d_hidden_multiplier))
        self.model = FTTransformer(n_cont_features=self.d_in, cat_cardinalities=[], d_out=1, n_blocks=self.n_blocks, d_block=d_block, attention_n_heads=self.attention_n_heads, attention_dropout=self.attention_dropout, ffn_d_hidden=ffn_hidden, ffn_d_hidden_multiplier=None, ffn_dropout=self.ffn_dropout, residual_dropout=self.residual_dropout).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        # SUPPRESS OUTPUT HERE
        with contextlib.redirect_stdout(io.StringIO()):
            epochs_run = train_trans(self.model, criterion, optimizer, self.n_epochs, train_loader, val_loader, EarlyStopping(self.patience, self.checkpoint_path), self.checkpoint_path) if val_loader and self.patience else train_trans_no_early_stopping(self.model, criterion, optimizer, self.n_epochs, train_loader)
        return epochs_run
    
    def predict(self, X):
        Xt = self._prepare(X)
        loader = self._loader(Xt, torch.zeros((Xt.size(0), 1)), shuffle=False)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loader: preds.append(self.model(xb, None).reshape(-1))
        return torch.cat(preds).cpu().numpy()
    
    def predict_with_uncertainty(self, X, train_loader, sample_size: int = None):
        mu = self.predict(X); self.model.eval()
        residuals = []
        with torch.no_grad():
            for xb, yb in train_loader: residuals.append(yb.view(-1) - self.model(xb, None).view(-1))
        return mu, torch.std(torch.cat(residuals), correction=1).item()


class FTTrans_Classifier(_TorchBase):
    def __init__(self, num_classes: int, n_blocks=2, d_block_multiplier=8, attention_n_heads=8, attention_dropout=0.5, ffn_d_hidden_multiplier=0.5, ffn_dropout=0.5, residual_dropout=0.5, learning_rate=1e-3, weight_decay=0.0, batch_size=32, patience=10, checkpoint_path="checkpoint_fttrans.pt", seed=None, n_epochs=100):
        super().__init__(batch_size=batch_size, classification=True, learning_rate=learning_rate, weight_decay=weight_decay, n_epochs=n_epochs, patience=patience, checkpoint_path=checkpoint_path, seed=seed)
        self.n_blocks = n_blocks; self.d_block_multiplier = d_block_multiplier; self.attention_n_heads = int(attention_n_heads); self.attention_dropout = attention_dropout; self.ffn_d_hidden_multiplier = ffn_d_hidden_multiplier; self.ffn_dropout = ffn_dropout; self.residual_dropout = residual_dropout; self.num_classes = num_classes; self.d_out = self.num_classes

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if self.seed is not None:
            torch.manual_seed(self.seed); torch.cuda.manual_seed_all(self.seed); np.random.seed(self.seed)
        train_loader, val_loader, _ = self.prepare_data(X_train, y_train, X_val, y_val)
        self.d_in = train_loader.dl.dataset.tensors[0].size(1)
        d_block = int(self.d_block_multiplier * self.attention_n_heads)
        ffn_hidden = max(2, int(d_block * self.ffn_d_hidden_multiplier))
        self.model = FTTransformer(n_cont_features=self.d_in, cat_cardinalities=[], d_out=self.d_out, n_blocks=self.n_blocks, d_block=d_block, attention_n_heads=self.attention_n_heads, attention_dropout=self.attention_dropout, ffn_d_hidden=ffn_hidden, ffn_d_hidden_multiplier=None, ffn_dropout=self.ffn_dropout, residual_dropout=self.residual_dropout).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss() if self.d_out == 1 else nn.CrossEntropyLoss()

        # SUPPRESS OUTPUT HERE
        with contextlib.redirect_stdout(io.StringIO()):
            epochs_run = train_trans(self.model, criterion, optimizer, self.n_epochs, train_loader, val_loader, EarlyStopping(self.patience, self.checkpoint_path), self.checkpoint_path) if val_loader and self.patience else train_trans_no_early_stopping(self.model, criterion, optimizer, self.n_epochs, train_loader)
        return epochs_run
    
    def predict_proba(self, X):
        Xt = self._prepare(X)
        loader = self._loader(Xt, torch.zeros((Xt.size(0), 1)), shuffle=False)
        self.model.eval()
        probs = []
        with torch.no_grad():
            for xb, _ in loader:
                logits = self.model(xb, None) 
                probs.append(torch.sigmoid(logits) if self.d_out == 1 else torch.softmax(logits, dim=1))
        return torch.cat(probs, dim=0).cpu().numpy().reshape(-1) if self.d_out == 1 else torch.cat(probs, dim=0).cpu().numpy()

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int).flatten() if self.d_out == 1 else np.argmax(self.predict_proba(X), axis=1)