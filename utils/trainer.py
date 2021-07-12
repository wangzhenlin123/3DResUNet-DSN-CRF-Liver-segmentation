import torch
import os
from tensorboardX import SummaryWriter
from time import time
from tqdm import tqdm
from utils.common import *
import numpy as np


class trainer(object):
    def __init__(self, model, optimizer, max_epochs, lr_scheduler, loss_function, comments,
                  train_dataloader, val_dataloader, alpha, ckpt_frequency, ckpt_alpha, verbose_train,
                 checkpoint_dir, device=torch.device('cuda:0')):
        self.model = model
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.epoch = 0
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.current_val_loss = 0.0
        self.current_val_dice = 0.0
        self.alpha = alpha
        self.ckpt_frequency = ckpt_frequency
        self.ckpt_alpha = ckpt_alpha
        self.verbose_train = verbose_train
        # self.verbose_val = verbose_val
        self.checkpoint_dir = checkpoint_dir
        self.comments = comments
        self.writer = SummaryWriter(comment=self.comments)
        self.device = device
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def train(self):
        out_file = open('..../epoch_loss_dice/lits17_test_four_six','w')
        print('epoch,trian_loss,train_dice,val_loss,val_dice',file=out_file)
        for self.epoch in range(self.max_epochs):
            current_lr = self.lr_scheduler.get_last_lr()
            print('Epoch：{}'.format(self.epoch))
            print('learning rate: {}'.format(current_lr[-1]))
            print('current_alpha:{:.3f}'.format(self.alpha))
            print('--------------------------------------------------------------------------------------')
            epoch_train_loss, epoch_train_dice = self.train_(self.train_dataloader)
            epoch_val_loss, epoch_val_dice = self.validate_(self.val_dataloader)
            print(self.epoch, epoch_train_loss, epoch_train_dice, epoch_val_loss, epoch_val_dice, sep=',', file=out_file)
            self.writer.add_scalar('Tr/Loss(end of epoch)', epoch_train_loss, self.epoch + 1)
            self.writer.add_scalar('Tr/Dice(end of epoch)', epoch_train_dice, self.epoch + 1)
            self.writer.add_scalar('Val/Loss(end of epoch)', epoch_val_loss, self.epoch + 1)
            self.writer.add_scalar('Val/Dice(end of epoch)', epoch_val_dice, self.epoch + 1)
            self.writer.add_scalars('Tr/Val/Loss',{'tr_loss':epoch_train_loss,
                                                   'val_loss':epoch_val_loss}, self.epoch + 1)
            self.writer.add_scalars('Tr/Val/Dice', {'tr_dice': epoch_train_dice,
                                                    'val_dice': epoch_val_dice}, self.epoch + 1)
            print('--------------------------------------------------------------------------------------')
            print('End of epoch:')
            print('train loss: {:.3f}, train_dice:{:.3f}'.format(epoch_train_loss, epoch_train_dice))
            print('val loss: {:.3f}, val_dice:{:.3f}'.format(epoch_val_loss, epoch_val_dice))
            print('--------------------------------------------------------------------------------------')
            self.lr_scheduler.step()

            if self.epoch % self.ckpt_frequency is 0 and self.epoch is not 0:
                checkpoint_name = os.path.join(self.checkpoint_dir,
                                           self.comments + 'epoch_{}_loss_{:.3f}.pth'.format(self.epoch, epoch_train_loss))
                torch.save(self.model.state_dict(), checkpoint_name)

            if self.epoch % self.ckpt_alpha is 0 and self.epoch is not 0:
                self.alpha *= 0.8
                print('after_alpha:{:.3f}'.format(self.alpha))
                print('--------------------------------------------------------------------------------------')

    def train_(self, data_loader):

        start_time = time()
        self.model.train()
        loss_total = 0.0
        train_dice_total = 0.0

        for i, (ct, seg) in enumerate(data_loader):

            seg = seg.to(self.device)
            # print(seg.shape)
            ct = ct.to(self.device)

            output = self.model(ct)

            loss1 = self.loss_function(output[0], seg)
            loss2 = self.loss_function(output[1], seg)
            loss3 = self.loss_function(output[2], seg)
            loss4 = self.loss_function(output[3], seg)

            # loss = self.loss_function(output, seg)

            loss = (loss1 + loss2 + loss3) * self.alpha + loss4

            seg = seg.reshape(1, 1, 48, 512, 512)
            dice1 = dice(output[0], seg, 0)
            dice2 = dice(output[1], seg, 0)
            dice3 = dice(output[2], seg, 0)
            dice4 = dice(output[3], seg, 0)
            # dice_tr = dice(output, seg, 0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # loss_total_output4 += loss4.item() * ct.size(0)
            loss_total += loss.item() * ct.size(0)
            # loss_total.append(loss4.item())
            train_dice_total += float(dice4)
            # train_dice_total += float(dice_tr)

            if i % self.verbose_train is 0:

                print('epoch:{}, step:{}, loss1:{:.3f}, loss2:{:.3f}, loss3:{:.3f}, loss4:{:.3f}, dice1:{:.3f}, dice2:{:.3f}, dice3:{:.3f}, dice4:{:.3f}, time:{:.3f} min'
                      .format(self.epoch, i, loss1.item(), loss2.item(), loss3.item(), loss4.item(),dice1, dice2, dice3, dice4,
                              (time() - start_time) / 60))
                # print(
                #     'epoch:{}, step:{}, loss:{:.3f}, tr_dice:{:.3f}, time:{:.3f} min'
                #     .format(self.epoch, i, loss.item(), dice_tr,(time() - start_time) / 60))


        # loss_output4 = loss_total_output4 / len(data_loader)
        # loss_ouptut_all = sum(loss_total) / len(loss_total)
        loss_ouptut_all = loss_total / data_loader.dataset.__len__()
        train_dice_total /= len(data_loader)
        return loss_ouptut_all, train_dice_total

    def validate_(self, data_loader):
        # self.model.eval()
        start_time = time()
        val_loss_total = 0
        val_dice_total = 0
        with torch.no_grad():
            for i, (ct, seg) in enumerate(data_loader):

                seg = seg.to(self.device)
                ct = ct.to(self.device)

                output = self.model(ct)
                # print(output.shape)
                # exit()
                loss_val1 = self.loss_function(output[0], seg)
                loss_val2 = self.loss_function(output[1], seg)
                loss_val3 = self.loss_function(output[2], seg)
                loss_val4 = self.loss_function(output[3], seg)
                # loss = self.loss_function(output, seg)

                loss = (loss_val1 + loss_val2 + loss_val3) * self.alpha + loss_val4

                seg = seg.reshape(1, 1, 48, 512, 512)
                dice1 = dice(output[0], seg, 0)
                dice2 = dice(output[1], seg, 0)
                dice3 = dice(output[2], seg, 0)
                dice4 = dice(output[3], seg, 0)
                # dice_val = dice(output, seg, 0)

                # val_dice = dice(output, seg.reshape(1, 1, 48, 512, 512), 0 )

                val_dice_total += float(dice4)
                # val_dice_total += float(dice_val)
                val_loss_total += loss.item() * ct.size(0)
                print('--------------------------------------------------------------------------------------')
                print(
                    'epoch:{}, step:{}, loss_val1:{:.3f}, loss_val2:{:.3f}, loss_val3:{:.3f}, loss_val4:{:.3f}, val_dice1:{:.3f}, val_dice2:{:.3f}, val_dice3:{:.3f}, val_dice4:{:.3f}, time:{:.3f} min'
                    .format(self.epoch, i, loss_val1.item(), loss_val2.item(), loss_val3.item(), loss_val4.item(), dice1, dice2, dice3,
                            dice4,
                            (time() - start_time) / 60))
                # print(
                #     'epoch:{}, step:{}, loss_val:{:.3f}, val_dice:{:.3f}, time:{:.3f} min'
                #         .format(self.epoch, i, loss.item(), dice_val,(time() - start_time) / 60))
                # print('--------------------------------------------------------------------------------------')
        loss_output = val_loss_total / data_loader.dataset.__len__()
        # loss_output = sum(val_loss_total) / len(val_loss_total)
        val_dice_total /= len(data_loader)
        self.model.train()
        return loss_output, val_dice_total







