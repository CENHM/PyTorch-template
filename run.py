import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.arguments import cfgs
from utils.utils import SET_SEED, LOAD_WEIGHT, SAVE_CHECKPOINT, LOAD_CHECKPOINT, VISUALIZE_TENSOR_SHAPE
from utils.utils import Logger
from utils.dataset import Dataset

from models.net import Net


def tests():
    global DEVICE, MODEL, CRITERION, DATASET, DATALOADER

    MODEL = LOAD_WEIGHT(MODEL, cfgs.checkpoint_path, cfgs.checkpoint_file)

    MODEL.eval()
    with torch.no_grad():
        loss = 0.
        acc, tot = 0, 0
        loader_iter = iter(DATALOADER)
        for batch_idx in range(len(DATALOADER)):
            x, y = next(loader_iter)
            x, y = x.to(DEVICE), y.to(DEVICE)

            predict = MODEL(x)

            loss += CRITERION(predict, y)
            prediction = torch.argmax(predict, 1)
            acc += (prediction == y).sum().item()
            tot += y.shape[0]

        log(f"TEST - loss: {loss / tot} - acc: {acc / tot}")
        LOG.SAVE_LOG()


def train(start_epoch=0):
    global DEVICE, MODEL, CRITERION, DATASET, DATALOADER, OPTIMIZER, LOG

    if cfgs.resume_test:
        start_epoch, OPTIMIZER, MODEL = LOAD_CHECKPOINT(OPTIMIZER, MODEL, cfgs.checkpoint_path, cfgs.checkpoint_file)

    MODEL.train()    
    for epoch in range(start_epoch + 1, cfgs.epoch + 1):
        loader_iter = iter(DATALOADER)
        for batch_idx in range(len(DATALOADER)):
            x, y = next(loader_iter)
            x, y = x.to(DEVICE), y.to(DEVICE)

            MODEL.zero_grad()
            predict = MODEL(x)
            loss = CRITERION(predict, y)
            loss.backward()
            OPTIMIZER.step()

            if batch_idx % 20 == 0:
                log(f"TRAIN - epoch: {epoch} - batch: {batch_idx + 1} / {len(DATALOADER) + 1} - loss: {loss}")
        SAVE_CHECKPOINT(epoch, OPTIMIZER, MODEL, cfgs.checkpoint_path)
        LOG.SAVE_LOG()
        print(f'Epoch {epoch} done.')


if __name__ == "__main__":
    VISUALIZE_TENSOR_SHAPE()
    SET_SEED()
    LOG = Logger(cfgs.testing, cfgs.checkpoint_path, cfgs.checkpoint_file, cfgs.result_path)
    log = LOG.WRITE

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL = Net().to(DEVICE)

    CRITERION = torch.nn.CrossEntropyLoss()

    if not cfgs.testing:
        print("training...")
        DATASET = Dataset(train=True)
        DATALOADER = DataLoader(DATASET, batch_size=cfgs.batch_size, shuffle=True)
        OPTIMIZER = optim.Adam(MODEL.parameters(), lr=cfgs.lr, weight_decay=cfgs.weight_decay)
        train()

    else:
        print("testing...")
        DATASET = Dataset(train=False)
        DATALOADER = DataLoader(DATASET, batch_size=cfgs.batch_size, shuffle=False)
        tests()

    print("end.")
        

    
