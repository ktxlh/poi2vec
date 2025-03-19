# %%
import numpy as np
from models import POI2VEC
import torch
import pandas as pd
import torch.nn as nn
from torch.optim import Adafactor, AdamW
from trajfm import get_train_val_test_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from copy import deepcopy
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Load raw POI data
checkin_file = "/home/Shared/foursquare/dataset_TSMC2014_NYC_preprocessed.csv"
dataset = pd.read_csv(checkin_file, encoding='ISO-8859-1')
dataset.columns = ["User ID", "Venue ID", "Category", "X", "Y", "Time"]

num_categories = dataset["Category"].nunique()
num_pois = dataset["Venue ID"].nunique()
print("Number of categories: ", num_categories)
print('Number of POIs: ', num_pois)

id2route = torch.from_numpy(np.load("./npy/id2route.npy"))
id2lr = torch.from_numpy(np.load("./npy/id2lr.npy"))
id2prob = torch.from_numpy(np.load("./npy/id2prob.npy")).float()

assert num_pois == len(id2route), "Number of POIs do not match"
assert num_pois == max(dataset['Venue ID']) + 1, "Number of POIs do not match"

pretrained_model = POI2VEC(
    poi_cnt=num_pois,
    id2route=id2route,
    id2lr=id2lr,
    id2prob=id2prob
)

pretrained_model_path = "/home/kate/poi2vec/model/p2v_model_20.pt"
pretrained_model.load_state_dict(torch.load(pretrained_model_path))

# %%
poi2cat = dataset[["Venue ID", "Category"]].drop_duplicates()
poi2cat = poi2cat.set_index("Venue ID").to_dict()["Category"]

# %%
_, _, _, (train_pois, val_pois, test_pois) = get_train_val_test_split(dataset)

# %%
class DataCollate:
    def __init__(self, poi2cat):
        self.poi2cat = poi2cat
        self.id2route = id2route.to(device)
        self.id2prob = id2prob.to(device)
    
    def __call__(self, poi_ids):
        cat_ids = list(map(lambda x: self.poi2cat[x], poi_ids))
        cat_ids = torch.tensor(cat_ids, device=device)
        poi_ids = torch.tensor(poi_ids, device=device)
        route_ids = self.id2route[poi_ids]
        route_probs = self.id2prob[poi_ids]
        return poi_ids, route_ids, cat_ids, route_probs
    
collator = DataCollate(poi2cat)

train_loader = torch.utils.data.DataLoader(train_pois, batch_size=128, shuffle=True, collate_fn=collator)
val_loader = torch.utils.data.DataLoader(val_pois, batch_size=128, shuffle=True, collate_fn=collator)
test_loader = torch.utils.data.DataLoader(test_pois, batch_size=128, shuffle=True, collate_fn=collator)

# %%
class PoiCatClassifier(nn.Module):
    def __init__(self, pretrained_model, num_categories, dim_hidden):
        super(PoiCatClassifier, self).__init__()
        num_routes, dim_model = pretrained_model.route_weight.weight.data.shape

        self.route_weight = deepcopy(pretrained_model.route_weight)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim_model, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, num_categories)
        )

    def forward(self, poi, route, route_probs):
    
        batch_size, num_routes, route_depth = route.shape
        route = self.route_weight(route.view(batch_size, -1)) # (batch_size, num_routes * route_depth, dim_model)
        route = route.view(batch_size, num_routes, route_depth, -1)
        
        route = route_probs.view(batch_size, num_routes, 1, 1) * route
        route = route.sum(dim=1) # (batch_size, route_depth, dim_model)
        route = route.mean(dim=1) # (batch_size, dim_model)        

        x = self.mlp(route) # (batch_size, num_categories)
        return x
       
classifier = PoiCatClassifier(pretrained_model, num_categories, dim_hidden=64)
classifier.to(device)

# %%
best_loss, best_model = float('inf'), None
best_epoch = 0
num_epochs = 30

criterion = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = Adafactor(classifier.parameters())

for epoch in range(num_epochs):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    classifier.train()
    preds = []
    truths = []
    for pois, routes, categories, route_probs in pbar:
        optimizer.zero_grad()
        output = classifier(pois, routes, route_probs)
        loss = criterion(output, categories)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1)
        preds += pred.tolist()
        truths += categories.tolist()

        accuracy = accuracy_score(truths, preds)
        pbar.set_postfix({"loss": loss.item(), "accuracy": accuracy})

    classifier.eval()
    val_loss = 0
    val_acc = []
    preds = []
    with torch.no_grad():
        for pois, routes, categories, route_probs in val_loader:
            output = classifier(pois, routes, route_probs)
            loss = criterion(output, categories)
            val_loss += loss.item()

            pred = output.argmax(dim=1)
            preds += pred.tolist()

            accuracy = accuracy_score(categories.tolist(), pred.tolist())
            val_acc.append(accuracy)


    val_acc = np.mean(val_acc)
    val_loss /= len(val_loader)
    ctr = Counter(preds)
    print(f"Epoch {epoch} - Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}")
    if val_loss < best_loss:
        print(f"New best model found at epoch {epoch} with loss {val_loss:.4f}")
        best_loss = val_loss
        best_epoch = best_epoch
        best_model = deepcopy(classifier.state_dict())

        # Save this model
        torch.save(best_model, "/home/kate/poi2vec/model/p2v_cat_classifier.pt")

# %%
classifier.load_state_dict(best_model)
classifier.eval()

# Test
test_loss = 0
y_true, y_pred = [], []
with torch.no_grad():
    for pois, routes, categories, route_probs in test_loader:
        output = classifier(pois, routes, route_probs)
        loss = criterion(output, categories)
        test_loss += loss.item()
        y_true.extend(categories.tolist())
        y_pred.extend(output.argmax(dim=1).tolist())

# Compute metrics
test_loss /= len(test_loader)
print(f"Test loss: {test_loss}")

# metrics: accuracy_score, f1_score, precision_score, recall_score
acc = accuracy_score(y_true, y_pred)
average = 'macro'
f1 = f1_score(y_true, y_pred, average=average)
precision = precision_score(y_true, y_pred, average=average, zero_division=0)
recall = recall_score(y_true, y_pred, average=average)
print("Test metrics:")
print("\t".join(["Accuracy", "Precision", "Recall", "F1"]))
print('\t'.join([str(x) for x in [acc, precision, recall, f1]]))

ctr = Counter(y_pred)
print("Predictions: ", [ctr[i] for i in range(num_categories)])

# %%



