# %%
import numpy as np
import pandas as pd
from datetime import datetime
import tqdm
from collections import Counter

checkin_file = "/home/Shared/foursquare/dataset_TSMC2014_NYC_preprocessed.csv"
df = pd.read_csv(checkin_file, encoding='ISO-8859-1')

df.columns = ["user", "poi", "Venue Category ID", "X", "Y", "time"]
df = df[['user', 'time', 'poi']]

# %%
# Sort df by user and time
df = df.sort_values(by=['user', 'time'])

maxlen_context = 32

# %%
from tqdm import tqdm

def split_data(user_context, user_target, train_ratio=0.98):
    train_context, train_target, valid_context, valid_target = [], [], [], []
    random_idx = np.random.permutation(len(user_context))
    user_context = np.array(user_context)
    user_target = np.array(user_target)
    train_idx = random_idx[:int(len(user_context)*train_ratio)]
    valid_idx = random_idx[int(len(user_context)*train_ratio):]
    train_context, train_target = user_context[train_idx], user_target[train_idx]
    valid_context, valid_target = user_context[valid_idx], user_target[valid_idx]
    return train_context.tolist(), train_target.tolist(), valid_context.tolist(), valid_target.tolist()

def process_user_data(df):
    train_context, train_target, valid_context, valid_target = [], [], [], []
    for user, group in tqdm(df.groupby('user')):
        user_context, user_target = [], []
        for target_idx, (_, row) in enumerate(group.iterrows()):
            time, poi = row['time'], row['poi']
            min_idx = max(target_idx - maxlen_context, 0)
            context = group.iloc[min_idx:target_idx]['poi'].tolist()
            if context:
                if len(context) < maxlen_context:
                    context += ([0]*(maxlen_context-len(context)))
                user_context.append(context)
                user_target.append(poi)
        
        split_results = split_data(user_context, user_target)
        train_context += split_results[0]
        train_target += split_results[1]
        valid_context += split_results[2]
        valid_target += split_results[3]
    
    return train_context, train_target, valid_context, valid_target

# Usage
train_context, train_target, valid_context, valid_target = process_user_data(df)

# %%
np.save('./npy/train_context.npy', train_context)
np.save('./npy/valid_context.npy', valid_context)
np.save('./npy/train_target.npy', train_target)
np.save('./npy/valid_target.npy', valid_target)


