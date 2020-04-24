from mlmodels.preprocess.generic import get_dataset_torch

data_pars = {
    'transform': False,
    'train_path': False,
    'test_path': False,
    'dataset': 'mlmodels.preprocess.generic:pandasDataset',
    # 'data_path': 'mlmodels/dataset/vision/',
    'train_batch_size': 16,
    'test_batch_size': 16,
    'train_path': 'mlmodels/dataset/text/ag_news_csv/',
    'test_path': 'mlmodels/dataset/text/ag_news_csv/',
    'train_filename': 'train.csv',
    'test_filename': 'test.csv',
    'coly': 0,
    'colX': 2,
    'no_header': True
    # 'transform_uri': 'mlmodels.preprocess.image:torch_transform_mnist'
}

train_loader, valid_loader = get_dataset_torch(data_pars)
for x, y in train_loader:
    print(x)
    print(y)
    break