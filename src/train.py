from typing import Optional, List, Any
import sys
import gc
from collections import namedtuple
import torch
from tqdm import tqdm

from utils import RunningAverage, iou_coef


def train(model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader,
          epochs: int,
          lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
          filename: str,
          accumulate_every_n_epochs: int = 1) -> List[Any]:
    """
    Train the model and evaluate every epoch

    Parameters
    ----------
    model (torch.nn.Module): pytorch neural network model
    optimizer (torch.optim.Optimizer): pytorch optimizer object
    criterion (torch.nn.Module): pytorch criterion that computes a gradient according to a given loss function
    train_loader (torch.utils.data.DataLoader): pytorch data loading iterable over the training dataset
    val_loader (torch.utils.data.DataLoader): pytorch data loading iterable over the validation dataset
    epochs (int): total number of epochs
    lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
    filename (str): string containing the filename to save the model and optimizer states to a disk file
    accumulate_every_n_epochs (int): epochs to accumulate gradient before updating the weights

    Returns
    -------
    history (List[EpochStats]): training history
    """
    device = torch.device('cuda')
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()
    EpochStats = namedtuple('EpochStats', 'epoch learning_rate train_loss val_loss val_jac time')
    history = []

    for e in range(epochs):
        loss_avg = RunningAverage()
        val_jac_avg = RunningAverage()
        val_loss_avg = RunningAverage()
        best_val_jac = 0.0

        torch.cuda.empty_cache()
        gc.collect()

        model.train()
        with tqdm(total=len(train_loader), leave=False, file=sys.stdout) as t:
            if lr_scheduler is not None:
                stats_current_lr = lr_scheduler.get_last_lr()[0]
            else:
                stats_current_lr = optimizer.param_groups[0]['lr']
            t.set_description(f'Epoch {e + 1}, LR {stats_current_lr:.6f}')

            for batch_n, batch_data in enumerate(train_loader):
                train_batch, labels_batch = batch_data['image'], batch_data['mask']
                train_batch, labels_batch = train_batch.to(device), labels_batch.to(device)

                with torch.autocast(device_type='cuda'):
                    output_batch = model(train_batch)
                    loss = criterion(output_batch, labels_batch)
                    loss_avg.update(loss.item())
                    loss /= accumulate_every_n_epochs

                scaler.scale(loss).backward()

                if (batch_n + 1) % accumulate_every_n_epochs == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                t.set_postfix({'stats': f'train_loss: {loss_avg():.4f}'})
                t.update()
                stats_time_elapsed = t.format_interval(t.format_dict['elapsed'])

        if lr_scheduler is not None:
            lr_scheduler.step()

        model.eval()
        with torch.no_grad():
            for batch_data in val_loader:
                val_batch, val_labels_batch = batch_data['image'], batch_data['mask']
                val_batch, val_labels_batch = val_batch.to(device), val_labels_batch.to(device)

                val_output_batch = model(val_batch)
                val_loss = criterion(val_output_batch, val_labels_batch)
                val_loss_avg.update(val_loss.item())

                val_predicted = torch.nn.Sigmoid()(val_output_batch)
                val_jac_batch = iou_coef(val_labels_batch, val_predicted)
                val_jac_avg.update(val_jac_batch.item())

        stats_epoch = EpochStats(epoch=e + 1,
                                 learning_rate=stats_current_lr,
                                 train_loss=loss_avg(),
                                 val_loss=val_loss_avg(),
                                 val_jac=val_jac_avg(),
                                 time=stats_time_elapsed)
        history.append(stats_epoch)

        print(
            f'Epoch {stats_epoch.epoch}. LR {stats_epoch.learning_rate:.6f}, train_loss: {stats_epoch.train_loss:.4f},'
            f' val_loss: {stats_epoch.val_loss:.4f}, val_jac: {stats_epoch.val_jac:.4f}, time: {stats_epoch.time}')

        if val_jac_avg() > best_val_jac:
            torch.save(model, filename)
            best_val_jac = val_jac_avg()

    return history
