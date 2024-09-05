import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.arguments import CFGS
from utils.utils import INITIALIZER, log, save_log
from utils.utils import LOAD_WEIGHT, SAVE_CHECKPOINT, LOAD_CHECKPOINT
from utils.dataset import Dataset

from models.net import Net


def tests():
    global DEVICE, MODEL, CRITERION, DATASET, DATALOADER
    MODEL = LOAD_WEIGHT(MODEL)

    len = len(DATALOADER)
    MODEL.eval()
    with torch.no_grad():
        loss = 0.
        acc, tot = 0, 0
        loader_iter = iter(DATALOADER)
        for batch_idx in range(len):
            data, label = next(loader_iter)
            data, label = data.to(DEVICE), label.to(DEVICE)
            predict = MODEL(data)
            loss += CRITERION(predict, label)
            prediction = torch.argmax(predict, 1)
            acc += (prediction == label).sum().item()
            tot += label.shape[0]
        log(f"TEST - loss: {loss / tot} - acc: {acc / tot}")
        save_log()


def train_one_epoch(epoch):
    global DEVICE, MODEL, CRITERION, DATASET, DATALOADER, OPTIMIZER

    len = len(DATALOADER)
    MODEL.train()  
    loader_iter = iter(DATALOADER)
    for batch_idx in range(len):
        data, label = next(loader_iter)
        data, label = data.to(DEVICE), label.to(DEVICE)
        MODEL.zero_grad()
        predict = MODEL(data)
        loss = CRITERION(predict, label)
        loss.backward()
        OPTIMIZER.step()
        if batch_idx % 50 == 0:
            log(f"TRAIN - epoch: {epoch} - batch: {batch_idx + 1} / {len} - loss: {loss}")


def train(start_epoch=0):
    global DEVICE, MODEL, CRITERION, DATASET, DATALOADER, OPTIMIZER
    if CFGS.resume_test:
        start_epoch, OPTIMIZER, MODEL = LOAD_CHECKPOINT(OPTIMIZER, MODEL)

    for epoch in range(start_epoch + 1, CFGS.epoch + 1):
        train_one_epoch(epoch)
        SAVE_CHECKPOINT(epoch, OPTIMIZER, MODEL)
        save_log()
        print(f'Epoch {epoch} done.')


if __name__ == "__main__":

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL = Net().to(DEVICE)

    CRITERION = torch.nn.CrossEntropyLoss()

    if not CFGS.testing:
        print("training...")
        DATASET = Dataset(train=True)
        DATALOADER = DataLoader(DATASET, batch_size=CFGS.batch_size, shuffle=True)
        OPTIMIZER = optim.Adam(MODEL.parameters(), lr=CFGS.learning_rate, weight_decay=CFGS.weight_decay)
        train()

    else:
        print("testing...")
        DATASET = Dataset(train=False)
        DATALOADER = DataLoader(DATASET, batch_size=CFGS.batch_size, shuffle=False)
        tests()

    print("done.")
        

    
