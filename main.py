import torch.nn as nn
from utils.trainer import trainer
import torch
from dataset.dataset import Dataset
from torch.utils.data import Dataset as dataset, DataLoader
from loss.Tversky import TverskyLoss
from models.ResUNet import ResUNet, init
import parameter as para
import torch.backends.cudnn as cudnn


if __name__ == '__main__':
    device = para.device
    cudnn.benchmark = para.cudnn_benchmark

    model = ResUNet(training=True).to(device)
    model.apply(init)
    # model = UNet().to(device)
    # model.apply(init)
    learning_rate = para.learning_rate
    weight_decay = para.weight_decay
    max_epochs = para.max_epochs
    batch_size = para.batch_size
    learning_rate_decay = para.learning_rate_decay
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, para.learning_rate_decay)
    loss = TverskyLoss().to(device)

    dataset_path = para.dataset_liver_path
    train_ds = Dataset(dataset_path, model='train')
    train_dl = DataLoader(train_ds, para.batch_size, shuffle=True, num_workers=para.num_workers, pin_memory=para.pin_memory)
    val_ds = Dataset(dataset_path, model='val')
    val_dl = DataLoader(val_ds, para.batch_size, shuffle=True, num_workers=para.num_workers, pin_memory=para.pin_memory)

    alpha = para.alpha
    checkpoints_dir = '/home/haishan/Data/dataPeiQing/PeiQing/liver07_segmentation/checkpoints/lits17_test/lits17_test'
    comments = 'lits17_test_'
    verbose_train = 1
    # verbose_val = 25
    ckpt_frequency = 5
    ckpt_alpha = 40

    # trainer = trainer(model, optimizer, max_epochs, lr_scheduler, loss, comments, train_dl, val_dl, alpha, ckpt_frequency, ckpt_alpha, verbose_train, verbose_val, checkpoints_dir, device)
    trainer = trainer(model, optimizer, max_epochs, lr_scheduler, loss, comments, train_dl, val_dl, alpha,
                      ckpt_frequency, ckpt_alpha, verbose_train, checkpoints_dir, device)
    trainer.train()

