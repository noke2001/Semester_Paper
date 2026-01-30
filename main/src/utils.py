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
        path="checkpoint.pt",
    ):
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
            if self.counter >= self.patience:
                self.early_stop = True

    def _save_checkpoint(self, val_loss, model):
        ensure_dir_exists(self.path)
        # Unwrap model if compiled (handles _orig_mod automatically)
        if hasattr(model, "_orig_mod"):
            state_dict = model._orig_mod.state_dict()
        else:
            state_dict = model.state_dict()

        temp_path = self.path + f".{os.getpid()}.tmp"
        try:
            torch.save(state_dict, temp_path)
            os.rename(temp_path, self.path)
            self.val_loss_min = val_loss
        except OSError:
            pass 


def train(
    model,
    criterion,
    optimizer,
    max_epochs,
    train_loader,
    val_loader=None,
    early_stopper: EarlyStopping = None,
    checkpoint_path=None,
    debug=False,
    val_check_interval=10
):
    if early_stopper and checkpoint_path:
        try: os.remove(checkpoint_path)
        except: pass

    # REMOVED: model.to(device) 
    # REASON: Model is already on GPU from Neural_models.py. 
    # Calling .to() on a torch.compile() object causes a crash.
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_ce = isinstance(criterion, torch.nn.CrossEntropyLoss)
    is_bce = isinstance(criterion, torch.nn.BCEWithLogitsLoss)

    for epoch in range(1, max_epochs + 1):
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            # Data is ALREADY on device via FastTensorDataLoader, 
            # but standard loaders might need .to(). 
            # We keep .to() for safety, it's a no-op if already on device.
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            outputs = model(X)

            if is_ce:
                loss = criterion(outputs, y.view(-1).long())
            elif is_bce:
                loss = criterion(outputs.view(-1), y.view(-1).float())
            else:
                loss = criterion(outputs.view(-1), y.view(-1))

            loss.backward()
            optimizer.step()

        if val_loader is not None and early_stopper is not None and (epoch % val_check_interval == 0):
            model.eval()
            val_loss = 0.0
            n = 0
            with torch.no_grad():
                for Xv, yv in val_loader:
                    Xv, yv = Xv.to(device, non_blocking=True), yv.to(device, non_blocking=True)
                    out = model(Xv)

                    if is_ce:
                        val_loss += criterion(out, yv.view(-1).long()).item()
                    elif is_bce:
                        val_loss += criterion(out.view(-1), yv.view(-1).float()).item()
                    else:
                        val_loss += criterion(out.view(-1), yv.view(-1)).item()
                    n += 1

            val_loss /= max(n, 1)
            early_stopper(val_loss, model)
            
            if early_stopper.early_stop:
                try:
                    state = torch.load(checkpoint_path)
                    if hasattr(model, "_orig_mod"):
                         model._orig_mod.load_state_dict(state)
                    else:
                         model.load_state_dict(state)
                except:
                    pass
                return epoch

    return max_epochs


def train_no_early_stopping(
    model, criterion, optimizer, max_epochs, train_loader, debug=False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # REMOVED: torch.compile and model.to(device) to prevent crashes
    
    is_ce = isinstance(criterion, torch.nn.CrossEntropyLoss)
    is_bce = isinstance(criterion, torch.nn.BCEWithLogitsLoss)

    for epoch in range(1, max_epochs + 1):
        model.train()
        for X,y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            outputs = model(X)

            if is_ce:
                loss = criterion(outputs, y.view(-1).long())
            elif is_bce:
                loss = criterion(outputs.view(-1), y.view(-1).float())
            else:
                loss = criterion(outputs.view(-1), y.view(-1))
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
    val_check_interval=10
):
    if early_stopper and checkpoint_path:
        try: os.remove(checkpoint_path)
        except: pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # REMOVED: torch.compile and model.to(device)

    is_ce = isinstance(criterion, torch.nn.CrossEntropyLoss)
    is_bce = isinstance(criterion, torch.nn.BCEWithLogitsLoss)

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            X, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            outputs = model(X, None)

            if is_ce:
                loss = criterion(outputs, y.view(-1).long())
            elif is_bce:
                loss = criterion(outputs.view(-1), y.view(-1).float())
            else:
                loss = criterion(outputs.view(-1), y.view(-1))

            loss.backward()
            optimizer.step()

        if val_loader is not None and early_stopper is not None and (epoch % val_check_interval == 0):
            model.eval()
            val_loss = 0.0
            n = 0
            with torch.no_grad():
                for vb in val_loader:
                    Xv, yv = vb[0].to(device, non_blocking=True), vb[1].to(device, non_blocking=True)
                    out = model(Xv, None)
                    
                    if is_ce:
                        val_loss += criterion(out, yv.view(-1).long()).item()
                    elif is_bce:
                        val_loss += criterion(out.view(-1), yv.view(-1).float()).item()
                    else:
                        val_loss += criterion(out.view(-1), yv.view(-1)).item()
                    n += 1

            val_loss /= max(n, 1)
            early_stopper(val_loss, model)
            
            if early_stopper.early_stop:
                try:
                    state = torch.load(checkpoint_path)
                    if hasattr(model, "_orig_mod"):
                         model._orig_mod.load_state_dict(state)
                    else:
                         model.load_state_dict(state)
                except:
                    pass
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
    # REMOVED: torch.compile and model.to(device)

    is_ce = isinstance(criterion, torch.nn.CrossEntropyLoss)
    is_bce = isinstance(criterion, torch.nn.BCEWithLogitsLoss)

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            X, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            outputs = model(X, None)

            if is_ce:
                loss = criterion(outputs, y.view(-1).long())
            elif is_bce:
                loss = criterion(outputs.view(-1), y.view(-1).float())
            else:
                loss = criterion(outputs.view(-1), y.view(-1))

            loss.backward()
            optimizer.step()

    return max_epochs