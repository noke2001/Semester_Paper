import os
import numpy as np
import torch
import inspect

# Utility to ensure checkpoint directories exist

def ensure_dir_exists(path):
    dirpath = os.path.dirname(path) or '.'
    os.makedirs(dirpath, exist_ok=True)

class EarlyStopping:
    def __init__(self, patience=40, verbose=False, delta=0.0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
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
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def _save_checkpoint(self, val_loss, model):
        ensure_dir_exists(self.path)
        tmp_path = self.path + '.tmp'
        try:
            torch.save(model.state_dict(), tmp_path)
            os.replace(tmp_path, self.path)
            if self.verbose:
                print(f"Validation loss decreased ({self.val_loss_min:.6f} -> {val_loss:.6f}). Saved model to {self.path}")
            self.val_loss_min = val_loss
        except Exception as e:
            print(f"Warning: failed atomic save ({e}), saving directly to {self.path}")
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss


def train(
    model,
    criterion,
    optimizer,
    max_epochs,
    train_loader,
    val_loader=None,
    early_stopper: EarlyStopping=None,
    checkpoint_path=None,
    debug=False
):
    """
    Generic training loop supporting various model signatures.
    """
    # Clean start: remove old checkpoint if provided
    if early_stopper and checkpoint_path:
        try: os.remove(checkpoint_path)
        except: pass

    # Determine how many inputs forward expects (excluding self)
    sig = inspect.signature(model.forward)
    n_params = len(sig.parameters) - 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(1, max_epochs+1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            X, y = batch[0].to(device), batch[1].to(device)
            if n_params == 1:
                outputs = model(X)
            elif n_params == 2:
                outputs = model(X, None)
            else:
                outputs = model(*X)

            outputs = outputs.view(-1)
            targets = y.view(-1)
            if debug and epoch == 1:
                print(f"Train shapes: X {X.shape}, out {outputs.shape}, yt {targets.shape}")
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if val_loader is not None and early_stopper is not None:
            model.eval()
            val_loss = 0.0
            n = 0
            with torch.no_grad():
                for vb in val_loader:
                    Xv, yv = vb[0].to(device), vb[1].to(device)
                    if n_params == 1:
                        out = model(Xv)
                    elif n_params == 2:
                        out = model(Xv, None)
                    else:
                        out = model(*Xv)
                    out = out.view(-1)
                    val_loss += criterion(out, yv.view(-1)).item()
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
                    filtered = {k: v for k, v in state.items() if k in md and v.shape == md[k].shape}
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(1, max_epochs+1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            X, y = batch[0].to(device), batch[1].to(device)
            outputs = model(X).view(-1)
            loss = criterion(outputs, y.view(-1))
            loss.backward()
            optimizer.step()
    return max_epochs


def train_trans(
    model, criterion, optimizer, max_epochs, train_loader,
    val_loader, early_stopper: EarlyStopping, checkpoint_path, debug=False
):
    return train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        max_epochs=max_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        early_stopper=early_stopper,
        checkpoint_path=checkpoint_path,
        debug=debug
    )


def train_trans_no_early_stopping(
    model, criterion, optimizer, max_epochs, train_loader, debug=False
):
    return train_no_early_stopping(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        max_epochs=max_epochs,
        train_loader=train_loader,
        debug=debug
    )
