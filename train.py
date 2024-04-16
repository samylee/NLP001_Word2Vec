import torch
import torch.nn as nn
import numpy as np
import os

from models.CBOW import CBOW
from utils.datasets import get_data_loader


class Trainer:
    def __init__(
            self,
            model,
            total_epochs,
            train_dataloader,
            val_dataloader,
            criterion,
            optimizer,
            lr_scheduler,
            print_freq=100,
            save_freq=5,
            save_path=None,
            device=None
    ):
        self.model = model
        self.total_epochs = total_epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.print_freq = print_freq
        self.save_freq = save_freq
        self.save_path = save_path
        self.device = device

        self.model.to(self.device)
        self.criterion.to(self.device)
        self.loss = {'train': [], 'val': []}

    def train(self):
        for epoch in range(self.total_epochs):
            self.train_epoch(epoch)
            self.val_epoch()
            print(
                'Epoch: {}/{}, Train Loss={:.5f}, Val Loss={:.5f}'.format(
                    epoch + 1,
                    self.total_epochs,
                    self.loss['train'][-1],
                    self.loss['val'][-1],
                )
            )

            self.lr_scheduler.step()

            if epoch % self.save_freq == 0:
                self.save_weights(epoch)

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = []

        for i, batch_data in enumerate(self.train_dataloader, 1):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())

            if i % self.print_freq == 0:
                print(
                    'Epoch: {}/{}, Iter: {}/{}, Train Loss={:.5f}'.format(
                        epoch,
                        self.total_epochs,
                        i,
                        len(self.train_dataloader),
                        loss.item()
                    )
                )

        epoch_loss = np.mean(running_loss)
        self.loss['train'].append(epoch_loss)

    def val_epoch(self):
        self.model.eval()
        running_loss = []

        with torch.no_grad():
            for i, batch_data in enumerate(self.val_dataloader, 1):
                inputs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss.append(loss.item())

        epoch_loss = np.mean(running_loss)
        self.loss['val'].append(epoch_loss)

    def save_weights(self, epoch):
        if epoch % self.save_freq == 0:
            model_name = 'checkpoint_{}.pt'.format(str(epoch))
            model_path = os.path.join(self.save_path, model_name)
            torch.save(self.model, model_path)


if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # train params
    base_lr = 0.025
    total_epochs = 100
    print_freq = 100
    save_freq = 5

    # data params
    data_dir = 'data/'
    train_batch_size = 96
    val_batch_size = 96

    # net params
    embed_dimension = 300

    # save path
    save_path = 'weights/'

    # get data loader
    print('load dataloader ...')
    train_dataloader, val_dataloader, vocab_size = get_data_loader(data_dir, train_batch_size, val_batch_size)
    print('vocab size: ', str(vocab_size))
    print('done!')

    # def model
    print('load cbow model ...')
    cbow = CBOW(vocab_size, embed_dimension)
    print('done!')

    # def criterion
    print('load criterion ...')
    criterion = nn.CrossEntropyLoss()
    print('done!')

    # def optimizer
    print('load optimizer ...')
    optimizer = torch.optim.Adam(cbow.parameters(), lr=base_lr)
    lr_lambda = lambda epoch: (total_epochs - epoch) / total_epochs
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=True)
    print('done!')

    trainer = Trainer(
        cbow,
        total_epochs,
        train_dataloader,
        val_dataloader,
        criterion,
        optimizer,
        lr_scheduler,
        print_freq,
        save_freq,
        save_path,
        device
    )

    print('training start!')
    trainer.train()
    print('training end!')
