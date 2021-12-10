from data.mab_dataset import MABDataset
from icf.icf import ICF

train_dataset = MABDataset('./data/ml-1m', 'train')
test_dataset = MABDataset('./data/ml-1m', 'test')

icf = ICF(train_dataset.n_users, train_dataset.n_items)

icf.load_params()

icf.interact_thompson(test_dataset)
