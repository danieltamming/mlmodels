from mlmodels.preprocess.generic import get_dataset_torch

data_pars = {
    'transform': False,
    'train_path': False,
    'test_path': False,
    'dataset': 'MNIST',
    # 'data_path': 'mlmodels/dataset/text/ag_news_csv/'
}

get_dataset_torch(data_pars)