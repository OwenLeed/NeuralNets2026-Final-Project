import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from model import GlucoseTransformer, TransformerConfig
from loss  import PinballLoss


@dataclass
class TrainingConfig:
    learning_rate  : float = 1e-3
    weight_decay   : float = 1e-4
    max_epochs     : int   = 100
    batch_size     : int   = 64

    lr_patience    : int   = 7
    lr_factor      : float = 0.5
    lr_min         : float = 1e-6

    early_stop_patience : int = 15

    checkpoint_dir  : str  = "checkpoints"
    checkpoint_name : str  = "best_model.pt"

    log_every_n_epochs : int = 1

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, self.checkpoint_name)


@dataclass
class TrainingHistory:
    train_loss         : List[float]        = field(default_factory=list)
    val_loss           : List[float]        = field(default_factory=list)
    learning_rates     : List[float]        = field(default_factory=list)
    val_loss_per_q     : List[List[float]]  = field(default_factory=list)
    val_coverage       : List[Dict]         = field(default_factory=list)
    best_epoch         : int                = 0
    best_val_loss      : float              = float('inf')
    total_train_time_s : float              = 0.0

    def summary(self):
        if not self.train_loss:
            print("No training history recorded.")
            return
        print(f"\nTraining Summary:")
        print(f"  Epochs trained    : {len(self.train_loss)}")
        print(f"  Best epoch        : {self.best_epoch + 1}")
        print(f"  Best val loss     : {self.best_val_loss:.6f}")
        print(f"  Final train loss  : {self.train_loss[-1]:.6f}")
        print(f"  Final val loss    : {self.val_loss[-1]:.6f}")
        print(f"  Final LR          : {self.learning_rates[-1]:.2e}")
        print(f"  Training time     : {self.total_train_time_s:.1f}s "
              f"({self.total_train_time_s/60:.1f}min)")

        if self.val_coverage:
            last_cov = self.val_coverage[-1]
            print(f"\n  Final validation coverage:")
            for name, vals in last_cov.items():
                print(f"    {name} interval: "
                      f"empirical={vals['empirical']:.3f}  "
                      f"target={vals['target']:.2f}  "
                      f"gap={vals['gap']:+.3f}")


class Trainer:

    def __init__(
        self,
        model          : GlucoseTransformer,
        train_config   : TrainingConfig,
        model_config   : TransformerConfig,
    ):
        self.model        = model
        self.train_config = train_config
        self.model_config = model_config
        self.device       = next(model.parameters()).device

        self.criterion = PinballLoss(model_config.quantiles).to(self.device)

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr           = train_config.learning_rate,
            weight_decay = train_config.weight_decay,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode     = 'min',
            factor   = train_config.lr_factor,
            patience = train_config.lr_patience,
            min_lr   = train_config.lr_min,
        )

        os.makedirs(train_config.checkpoint_dir, exist_ok=True)

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in loader:
            features = batch['features'].to(self.device)
            mask     = batch['attention_mask'].to(self.device)
            targets  = batch['target'].to(self.device)

            self.optimizer.zero_grad()

            predictions = self.model(features, mask)
            loss        = self.criterion(predictions, targets)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        return total_loss / n_batches

    def _val_epoch(
        self,
        loader: DataLoader,
    ) -> Tuple[float, List[float], Dict]:
        self.model.eval()
        total_loss   = 0.0
        n_batches    = 0
        all_preds    = []
        all_targets  = []

        with torch.no_grad():
            for batch in loader:
                features = batch['features'].to(self.device)
                mask     = batch['attention_mask'].to(self.device)
                targets  = batch['target'].to(self.device)

                predictions = self.model(features, mask)
                loss        = self.criterion(predictions, targets)

                total_loss  += loss.item()
                n_batches   += 1
                all_preds.append(predictions)
                all_targets.append(targets)

        all_preds   = torch.cat(all_preds,   dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        mean_loss   = total_loss / n_batches
        per_q_loss  = self.criterion.per_quantile_loss(
            all_preds, all_targets
        ).tolist()
        coverage    = self.criterion.coverage(all_preds, all_targets)

        return mean_loss, per_q_loss, coverage

    def fit(
        self,
        train_loader : DataLoader,
        val_loader   : DataLoader,
    ) -> TrainingHistory:
        history        = TrainingHistory()
        patience_count = 0
        t_start        = time.time()

        print(f"Training on {self.device}")
        print(f"Train batches/epoch : {len(train_loader)}")
        print(f"Val   batches/epoch : {len(val_loader)}")
        print(f"Max epochs          : {self.train_config.max_epochs}")
        print(f"Early stop patience : {self.train_config.early_stop_patience}")
        print()
        print(f"{'Epoch':>6}  {'Train':>10}  {'Val':>10}  "
              f"{'90% Cov':>9}  {'50% Cov':>9}  {'LR':>10}  {'':>6}")
        print("-" * 70)

        for epoch in range(self.train_config.max_epochs):
            train_loss = self._train_epoch(train_loader)
            val_loss, per_q_loss, coverage = self._val_epoch(val_loader)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)

            history.train_loss.append(train_loss)
            history.val_loss.append(val_loss)
            history.learning_rates.append(current_lr)
            history.val_loss_per_q.append(per_q_loss)
            history.val_coverage.append(coverage)

            cov_90 = coverage.get('90%', {}).get('empirical', float('nan'))
            cov_50 = coverage.get('50%', {}).get('empirical', float('nan'))

            if val_loss < history.best_val_loss:
                history.best_val_loss = val_loss
                history.best_epoch    = epoch
                patience_count        = 0
                flag                  = "<-- best"
                torch.save({
                    'epoch'        : epoch,
                    'model_state'  : self.model.state_dict(),
                    'optimizer'    : self.optimizer.state_dict(),
                    'val_loss'     : val_loss,
                    'model_config' : self.model_config,
                    'train_config' : self.train_config,
                }, self.train_config.checkpoint_path)
            else:
                patience_count += 1
                flag = (f"patience {patience_count}/"
                        f"{self.train_config.early_stop_patience}")

            if (epoch + 1) % self.train_config.log_every_n_epochs == 0:
                print(
                    f"{epoch+1:>6}  "
                    f"{train_loss:>10.6f}  "
                    f"{val_loss:>10.6f}  "
                    f"{cov_90:>9.3f}  "
                    f"{cov_50:>9.3f}  "
                    f"{current_lr:>10.2e}  "
                    f"{flag}"
                )

            if patience_count >= self.train_config.early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}.")
                print(f"No improvement for "
                      f"{self.train_config.early_stop_patience} epochs.")
                break

        checkpoint = torch.load(
            self.train_config.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint['model_state'])
        print(f"\nRestored best model from epoch "
              f"{history.best_epoch + 1}.")

        history.total_train_time_s = time.time() - t_start
        history.summary()

        return history
