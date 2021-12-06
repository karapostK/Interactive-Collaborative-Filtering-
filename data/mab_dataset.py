import os

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co


class MABDataset(Dataset):

    def __init__(self, data_path: str, split_set: str):
        assert split_set in ['train', 'test'], f'<{split_set}> is not a valid value for split set!'

        self.data_path = data_path
        self.split_set = split_set

        self.n_users = None
        self.n_items = None

        self.lhs = None

        self._load_data()

    def _load_data(self):
        print('Loading Data')

        user_ids = pd.read_csv(os.path.join(self.data_path, 'user_ids.csv'))
        item_ids = pd.read_csv(os.path.join(self.data_path, 'item_ids.csv'))

        self.n_users = len(user_ids)
        self.n_items = len(item_ids)

        self.lhs = pd.read_csv(os.path.join(self.data_path, f'listening_history_{self.split_set}.csv'))

    def __len__(self):
        return len(self.lhs)

    def __getitem__(self, index) -> T_co:
        entry = self.lhs.iloc[index]
        return entry.user_id, entry.item_id


def get_mab_dataloader(data_path: str, split_set: str, **loader_params) -> DataLoader:
    mab_dataset = MABDataset(data_path, split_set)

    return DataLoader(mab_dataset, **loader_params)
