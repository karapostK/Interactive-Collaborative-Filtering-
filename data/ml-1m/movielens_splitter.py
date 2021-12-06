import argparse
import os

import pandas as pd

INF_STR = "{:10d} entries {:7d} users {:7d} items for {}"

parser = argparse.ArgumentParser()

parser.add_argument('--listening_history_path', '-lh', type=str,
                    help="Path to 'ratings.dat' of the Movielens1M dataset.")
parser.add_argument('--saving_path', '-s', type=str, help="Path where to save the split data. Default to './'",
                    default='./')

args = parser.parse_args()

listening_history_path = args.listening_history_path
saving_path = args.saving_path

ratings_path = os.path.join(listening_history_path, 'ratings.dat')

lhs = pd.read_csv(ratings_path, sep='::', names=['user', 'item', 'rating', 'timestamp'])

print(INF_STR.format(len(lhs), lhs.user.nunique(), lhs.item.nunique(), 'Original Data'))

# We keep only ratings above 3.5
lhs = lhs[lhs.rating >= 3.5]
print(INF_STR.format(len(lhs), lhs.user.nunique(), lhs.item.nunique(), 'Only Positive Interactions (>= 3.5)'))

# Creating simple integer indexes used for sparse matrices
user_ids = lhs.user.drop_duplicates().reset_index(drop=True)
item_ids = lhs.item.drop_duplicates().reset_index(drop=True)
user_ids.index.name = 'user_id'
item_ids.index.name = 'item_id'
user_ids = user_ids.reset_index()
item_ids = item_ids.reset_index()
lhs = lhs.merge(user_ids).merge(item_ids)

print('Splitting the data - user-wise')

user_tr = user_ids.user_id.sample(frac=0.8, replace=False, random_state=42)
user_te = user_ids[~user_ids.user_id.isin(set(user_tr))].user_id

train_data = lhs[lhs.user_id.isin(set(user_tr))].sort_values('timestamp')
test_data = lhs[lhs.user_id.isin(set(user_te))].sort_values('timestamp')

print(INF_STR.format(len(train_data), train_data.user.nunique(), train_data.item.nunique(), 'Train Data'))
print(INF_STR.format(len(test_data), test_data.user.nunique(), test_data.item.nunique(), 'Test Data'))

# Saving locally

print('Saving data to {}'.format(saving_path))

train_data.to_csv(os.path.join(saving_path, 'listening_history_train.csv'), index=False)
test_data.to_csv(os.path.join(saving_path, 'listening_history_test.csv'), index=False)

user_ids.to_csv(os.path.join(saving_path, 'user_ids.csv'), index=False)
item_ids.to_csv(os.path.join(saving_path, 'item_ids.csv'), index=False)
