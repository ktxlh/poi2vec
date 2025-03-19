import gc
import time
from tqdm import tqdm
import torch
import torch.nn as nn

import config
import dataset
import models

# Type Parameters
ltype = config.ltype
ftype = config.ftype

def print_score(batches):
    batch_loss = 0.0  # hit count
    for i, batch in enumerate(batches):
        context_batch, target_batch = zip(*batch) 
        batch_loss += run(context_batch, target_batch)
    print("Validation Error :", batch_loss / (i + 1), time.ctime())
    return batch_loss / (i + 1)

##############################################################################################
def run(context, target):
    optimizer.zero_grad()

    context = torch.stack(context).type(config.ltype)
    target = torch.stack(target).type(config.ltype)

    # POI2VEC
    loss = p2v_model(context, target)

    loss.backward()
    optimizer.step()
    gc.collect()

    return loss.item()

##############################################################################################
##############################################################################################
if __name__ == "__main__":
    # Data Preparation
    data = dataset.Data()
    poi_cnt = data.load()

    # Model Preparation
    p2v_model = models.POI2VEC(poi_cnt, data.id2route, data.id2lr, data.id2prob).cuda()
    loss_model = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adafactor(p2v_model.parameters())

    p2v_model.to(config.device)
    best_loss = float('inf')
    best_epoch = 0

    for i in range(config.num_epochs):  # Changed xrange to range
        # Training
        batch_loss = 0.0
        train_batches = data.train_batch_iter(config.batch_size)
        total = len(data.target_train) // config.batch_size
        pbar = tqdm(train_batches, total=total, desc="Epoch #{:d}".format(i + 1))
        for j, train_batch in enumerate(pbar):
            context_batch, target_batch = zip(*train_batch) 
            batch_loss += run(context_batch, target_batch)
            pbar.set_postfix({'batch_loss': batch_loss / (j + 1)})
        pbar.close()

        # Validation 
        if (i + 1) % config.evaluate_every == 0:
            print("==================================================================================")
            print("Evaluation at epoch #{:d}: ".format(i + 1))
            p2v_model.eval()
            valid_batches = data.valid_batch_iter(config.batch_size)
            val_loss = print_score(valid_batches)
            p2v_model.train()

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = i + 1
                torch.save(p2v_model.state_dict(), f"./model/p2v_model_{best_epoch}.pt")

            if i + 1 - best_epoch >= config.patience:
                print("Early stopping at epoch #{:d}".format(i + 1))
                break
