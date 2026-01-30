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
        with open(self.path, 'wb') as f:
            try:              
                torch.save(model.state_dict(), f, _use_new_zipfile_serialization=False)
            except TypeError:
                torch.save(model.state_dict(), f)
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
        for X, y in train_loader:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            outputs = model(X)

            C = outputs.shape[1] if outputs.dim() == 2 else 1

            if C >= 2 and isinstance(criterion, torch.nn.CrossEntropyLoss):
                loss = criterion(outputs, y.view(-1).long())

            elif C == 1 and isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                logits  = outputs.view(-1)
                targets = y.view(-1).float()
                loss    = criterion(logits, targets)

            else:
                out_flat = outputs.view(-1)
                y_flat   = y.view(-1)
                loss     = criterion(out_flat, y_flat)

            loss.backward()
            optimizer.step()

        if val_loader is not None and early_stopper is not None:
            model.eval()
            val_loss = 0.0
            n = 0
            with torch.no_grad():
                for Xv, yv in val_loader:
                    Xv, yv = Xv.to(device), yv.to(device)
                    out = model(Xv)
                    C = outputs.shape[1] if outputs.dim() == 2 else 1


                    if C >= 2 and isinstance(criterion, torch.nn.CrossEntropyLoss):
                        val_loss += criterion(out, yv.view(-1).long()).item()
                    elif C == 1 and isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                        val_loss += criterion(out.view(-1), yv.view(-1).float()).item()
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
                # reload best checkpoint if available
                try:
                    state = torch.load(checkpoint_path)
                    md = model.state_dict()
                    filtered = {
                        k: v for k, v in state.items()
                        if k in md and v.shape == md[k].shape
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
            C = outputs.shape[1] if outputs.dim() == 2 else 1

            if C >= 2 and isinstance(criterion, torch.nn.CrossEntropyLoss):
                loss = criterion(outputs, y.view(-1).long())

            elif C ==1 and outputs.size(1) == 1 and isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                logits  = outputs.view(-1)
                targets = y.view(-1).float()
                loss    = criterion(logits, targets)

            else:
                out_flat = outputs.view(-1)
                y_flat   = y.view(-1)
                loss     = criterion(out_flat, y_flat)
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
            C = outputs.shape[1] if outputs.dim() == 2 else 1

            if  isinstance(criterion, torch.nn.CrossEntropyLoss):
                flat_out = outputs                 # shape (N, C)
                flat_y   = y.view(-1).long()       # shape (N,)

            elif  isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                flat_out = outputs.view(-1)        # shape (N,)
                flat_y   = y.view(-1).float()      # shape (N,)

            else:
                flat_out = outputs.view(-1)
                flat_y   = y.view(-1)

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
                    
                    C = out.shape[1] if outputs.dim() == 2 else 1

                    if  isinstance(criterion, torch.nn.CrossEntropyLoss):
                        out_flat, yv_flat = out, yv.view(-1).long()
                    elif  isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                        out_flat, yv_flat = out.view(-1), yv.view(-1).float()
                    else:
                        out_flat, yv_flat = out.view(-1), yv.view(-1)

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
                        k: v for k, v in state.items()
                        if k in md and v.shape == md[k].shape
                    }
                    md.update(filtered)
                    model.load_state_dict(md)
                except Exception as e:
                    print(f"Warning: could not load checkpoint: {e}")
                return epoch

    return max_epochs


def train_trans_no_early_stopping(
    model,
    criterion,
    optimizer,
    max_epochs,
    train_loader,
    debug=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(1, max_epochs + 1):
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)

            outputs = model(X, None)


            if outputs.dim() == 2 and outputs.size(1) == 1:
                y = y - 1
                flat_out = outputs.view(-1)  
                flat_y   = y.view(-1).float()
                loss     = torch.nn.BCEWithLogitsLoss()(flat_out, flat_y)

            elif isinstance(criterion, torch.nn.CrossEntropyLoss):
                flat_out = outputs
                flat_y   = y.view(-1).long()
                loss     = criterion(flat_out, flat_y)

            else:
                flat_out = outputs.view(-1)
                flat_y   = y.view(-1)
                loss     = criterion(flat_out, flat_y)

            loss = criterion(flat_out, flat_y)
            loss.backward()
            optimizer.step()

    return max_epochs