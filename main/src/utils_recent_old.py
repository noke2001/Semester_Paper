import os
import numpy as np
import torch
import inspect
import time

# Utility to ensure checkpoint directories exist
def ensure_dir_exists(path):
    dirpath = os.path.dirname(path) or "."
    os.makedirs(dirpath, exist_ok=True)


class EarlyStopping:
    def __init__(
        self,
        patience=40,
        verbose=False,
        delta=0.0,
        path="checkpoint.pt",  # Now an instance variable
    ):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path  # Store the path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self._save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True

    def _save_checkpoint(self, val_loss, model):
        ensure_dir_exists(self.path)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



def train(
    model,
    criterion,
    optimizer,
    max_epochs,
    train_loader,
    val_loader=None,
    early_stopper: EarlyStopping = None,
    checkpoint_path=None,  # Accept the path
    debug=False,
):
    """
    Generic training loop.
    """
    if early_stopper and checkpoint_path:
        try:
            os.remove(checkpoint_path)
        except:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(1, max_epochs + 1):
        model.train()
        for X,y in train_loader:
            optimizer.zero_grad()
            X, y    = X.to(device), y.to(device)
            outputs = model(X)

            if outputs.dim() == 2 and  isinstance(criterion, torch.nn.CrossEntropyLoss):
                loss = criterion(outputs, y.view(-1).long())
            else:
                out_flat = outputs.view(-1)
                y_flat   = y.view(-1)   
                loss = criterion(out_flat, y_flat)
            loss.backward()
            optimizer.step()

        if val_loader is not None and early_stopper is not None:
            model.eval()
            val_loss = 0.0
            n = 0
            with torch.no_grad():
                for vb in val_loader:
                    Xv, yv = vb[0].to(device), vb[1].to(device)
                    out = model(Xv)
                    if out.dim() == 2 and isinstance(criterion, torch.nn.CrossEntropyLoss):
                        val_loss += criterion(out, yv.view(-1).long()).item()
                    else:
                        val_loss += criterion(out.view(-1), yv.view(-1)).item()
                    n += 1
            val_loss /= max(n, 1)
            if debug:
                print(f"Epoch {epoch} val_loss={val_loss:.6f}")
            early_stopper(val_loss, model)
            if early_stopper.early_stop:
                if debug:
                    print(f"Early stopping at epoch {epoch}")
                try:
                    state = torch.load(checkpoint_path)
                    md = model.state_dict()
                    filtered = {
                        k: v for k, v in state.items() if k in md and v.shape == md[k].shape
                    }
                    md.update(filtered)
                    model.load_state_dict(md)
                except Exception as e:
                    print(f"Warning: could not load checkpoint: {e}")
                return epoch
    return max_epochs



def train_no_early_stopping(
    model, criterion, optimizer, max_epochs, train_loader, debug=False
):
    """
    Training loop without early stopping.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(1, max_epochs + 1):
        model.train()
        for X,y in train_loader:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            out_flat = outputs.view(-1)
            y_flat = y.view(-1)
            loss = criterion(out_flat, y_flat)
            loss.backward()
            optimizer.step()
    return max_epochs



def train_trans(
    model,
    criterion,
    optimizer,
    max_epochs,
    train_loader,
    val_loader,
    early_stopper: EarlyStopping,
    checkpoint_path,
    debug=False,
):
    if early_stopper and checkpoint_path:
        try:
            os.remove(checkpoint_path)
        except:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            X, y = batch[0].to(device), batch[1].to(device)
            outputs = model(X, None)
            if outputs.dim() == 1 or (outputs.dim() == 2 and outputs.size(1) == 1):
                flat_out = outputs.view(-1)
                flat_y   = y.view(-1)
            else:   
                flat_out = outputs
                flat_y   = y.view(-1).long()
            loss = criterion(flat_out, flat_y)
            loss.backward()
            optimizer.step()

        if val_loader is not None and early_stopper is not None:
            model.eval()
            val_loss = 0.0
            n = 0
            with torch.no_grad():
                for vb in val_loader:
                    Xv, yv = vb[0].to(device), vb[1].to(device)
                    out = model(Xv, None)
                    if out.dim() == 1 or (out.dim() == 2 and out.size(1) == 1):
                        out_flat = out.view(-1)
                        yv_flat = yv.view(-1)
                    else:
                        out_flat = out
                        yv_flat = yv.view(-1).long()
                    val_loss += criterion(out_flat, yv_flat).item()
                    n += 1
            val_loss /= max(n, 1)
            if debug:
                print(f"Epoch {epoch} val_loss={val_loss:.6f}")
            early_stopper(val_loss, model)
            if early_stopper.early_stop:
                if debug:
                    print(f"Early stopping at epoch {epoch}")
                try:
                    state = torch.load(checkpoint_path)
                    md = model.state_dict()
                    filtered = {
                        k: v for k, v in state.items() if k in md and v.shape == md[k].shape
                    }
                    md.update(filtered)
                    model.load_state_dict(md)
                except Exception as e:
                    print(f"Warning: could not load checkpoint: {e}")
                return epoch
    return max_epochs


def train_trans_no_early_stopping(
    model, criterion, optimizer, max_epochs, train_loader, debug=False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(1, max_epochs + 1):
        model.train()
        for X,y in train_loader:
            optimizer.zero_grad()
            X, y    = X.to(device), y.to(device)
            outputs = model(X, None)

            if outputs.dim() == 1 or outputs.size(1) == 1:
                flat_out = outputs.view(-1)
                flat_y   = y.view(-1)
            else:
                flat_out = outputs
                flat_y   = y.view(-1).long()  
            loss = criterion(flat_out, flat_y)
            loss.backward()
            optimizer.step()
    return max_epochs

