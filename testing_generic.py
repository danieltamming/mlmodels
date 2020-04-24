from mlmodels.preprocess.generic import get_dataset_torch

data_pars = {
    'transform': False,
    'train_path': False,
    'test_path': False,
    'dataset': 'MNIST',
    'data_path': 'mlmodels/dataset/vision/',
    'train_batch_size': 16,
    'test_batch_size': 16,
}

train_loader, valid_loader = get_dataset_torch(data_pars)
for x, y in train_loader:
    print(x.shape, y.shape)
    break