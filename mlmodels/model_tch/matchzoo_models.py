  
# -*- coding: utf-8 -*-
"""
https://github.com/NTMC-Community/MatchZoo-py/tree/master/tutorials
https://matchzoo.readthedocs.io/en/master/model_reference.html

https://github.com/NTMC-Community/MatchZoo-py/blob/master/tutorials/classification/esim.ipynb

"""
import os, json
import importlib
import torch
import matchzoo as mz
import numpy as np
import pandas as pd
from mlmodels.util import os_package_root_path, log, path_norm, get_model_uri, path_norm_dict

MODEL_URI = get_model_uri(__file__)

MODELS = {
    'DRMM' : mz.models.DRMM,
    'DRMMTKS' : mz.models.DRMMTKS, 
    'ARC-I' : mz.models.ArcI,
    'ARC-II' : mz.models.ArcII, 
    'DSSM' : mz.models.DSSM, 
    'CDSSM': mz.models.CDSSM, 
    'MatchLSTM' : mz.models.MatchLSTM,
    'DUET' : mz.models.DUET,
    'KNRM' : mz.models.KNRM,
    'ConvKNRM' : mz.models.ConvKNRM, 
    'ESIM' : mz.models.ESIM,
    'BiMPM' : mz.models.BiMPM,
    'MatchPyramid' : mz.models.MatchPyramid, 
    'Match-SRNN' : mz.models.MatchSRNN,
    'aNMM' : mz.models.aNMM,
    'HBMP' : mz.models.HBMP,
    'BERT' : mz.models.Bert
}

TASKS = {
    'ranking' : mz.tasks.Ranking,
    'classification' : mz.tasks.Classification,
}

METRICS = {
    'NormalizedDiscountedCumulativeGain' : mz.metrics.NormalizedDiscountedCumulativeGain,
    'MeanAveragePrecision' : mz.metrics.MeanAveragePrecision,
    'acc' : 'acc'
}

LOSSES = {
    'RankHingeLoss' : mz.losses.RankHingeLoss,
    'RankCrossEntropyLoss' : mz.losses.RankCrossEntropyLoss
}

from pytorch_transformers import AdamW
from torch.optim import Adadelta
OPTIMIZERS = {
    'ADAMW' : lambda prm, cp : AdamW(prm, lr=cp["lr"], betas=(cp["beta1"],cp["beta2"]), eps=cp["eps"]),
    'ADADELTA' : lambda prm, cp : Adadelta(prm, lr=cp["lr"], rho=cp["rho"], eps=cp["eps"], weight_decay=cp["weight_decay"])
}

CALLBACKS = {
    'PADDING' : lambda mn : MODELS[mn].get_default_padding_callback()
}

def get_task(model_pars):
    _task = model_pars['task']
    assert _task in TASKS.keys()
    if _task == "ranking":
        _loss = list(model_pars["loss"].keys())[0]
        _loss_params = model_pars["loss"][_loss]
        if _loss == 'RankHingeLoss':
            loss =  LOSSES[_loss]()
        elif _loss == 'RankCrossEntropyLoss':
            loss =  LOSSES[_loss](num_neg=_loss_params["num_neg"])
        task = mz.tasks.Ranking(losses=loss)
    elif _task == "classification" :
        task = mz.tasks.Classification(num_classes=model_pars["num_classes"])
    else:
        raise Exception(f"No support task {task} yet")

    _metrics = model_pars['metrics']
    task.metrics = []
    for metric in _metrics.keys():
        metric_params = _metrics[metric]
        # Find a better way later to apply params for metric, for now hardcode.
        if metric == 'NormalizedDiscountedCumulativeGain' and metric_params != {}:
            task.metrics.append(METRICS[metric](k=metric_params["k"]))
        elif metric in METRICS:
            task.metrics.append(METRICS[metric]())
        else:
            raise Exception(f"No support of metric {metric} yet")
    return task

def get_glove_embedding_matrix(term_index, dimension):
    glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=dimension)
    embedding_matrix = glove_embedding.build_matrix(term_index)
    l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
    embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]
    return embedding_matrix

def get_data_loader(model_name, preprocessor, preprocess_pars, raw_data):
    if "transform" in preprocess_pars:
        pack_processed = preprocessor.transform(raw_data)
    elif "fit_transform" in preprocess_pars:
        pack_processed = preprocessor.fit_transform(raw_data)

    mode = preprocess_pars["mode"] if "mode" in preprocess_pars else "point"
    num_dup = preprocess_pars["num_dup"] if "num_dup" in preprocess_pars else 1
    num_neg = preprocess_pars["num_neg"] if "num_neg" in preprocess_pars else 1
    dataset_callback = preprocess_pars["dataset_callback"] if "dataset_callback" in preprocess_pars else None
    glove_embedding_matrix_dim = preprocess_pars["glove_embedding_matrix_dim"] if "glove_embedding_matrix_dim" in preprocess_pars else None
    if glove_embedding_matrix_dim:
        # Make sure you've transformed data before generating glove embedding,
        # else, term_index would be 0 and embedding matrix would be None.
        term_index = preprocessor.context['vocab_unit'].state['term_index']
        embedding_matrix = get_glove_embedding_matrix(term_index, glove_embedding_matrix_dim)

    if dataset_callback == "HISTOGRAM":
        # For now, hardcode callback. Hard to generalize
        dataset_callback = [mz.dataloader.callbacks.Histogram(
            embedding_matrix, bin_size=30, hist_mode='LCH'
        )]

    resample = preprocess_pars["resample"] if "resample" in preprocess_pars else None
    sort = preprocess_pars["sort"] if "sort" in preprocess_pars else None
    batch_size = preprocess_pars["batch_size"] if "batch_size" in preprocess_pars else 1
    dataset = mz.dataloader.Dataset(
        data_pack=pack_processed,
        mode=mode,
        num_dup=num_dup,
        num_neg=num_neg,
        batch_size=batch_size,
        resample=resample,
        sort=sort,
        callbacks=dataset_callback
    )

    stage = preprocess_pars["stage"] if "stage" in preprocess_pars else None
    dataloader_callback = preprocess_pars["dataloader_callback"] if "dataloader_callback" in preprocess_pars else None
    dataloader_callback = CALLBACKS[dataloader_callback](model_name)
    dataloader = mz.dataloader.DataLoader(
        device='cpu',
        dataset=dataset,
        stage=stage,
        callback=dataloader_callback
    )
    return dataloader

def update_model_param(params, model, task, preprocessor):
    model.params['task'] = task
    glove_embedding_matrix_dim = params["glove_embedding_matrix_dim"] if "glove_embedding_matrix_dim" in params else None

    if glove_embedding_matrix_dim:
        term_index = preprocessor.context['vocab_unit'].state['term_index']
        embedding_matrix = get_glove_embedding_matrix(term_index, glove_embedding_matrix_dim)
        model.params['embedding'] = embedding_matrix
        # Remove those entried in JSON which not directly feeded to model as params
        del params["glove_embedding_matrix_dim"]

    # Feed rest all params directly to the model
    for key, value in params.items():
        model.params[key] = value

def get_config_file():
    return os.path.join(os_package_root_path(__file__, 1), 'config', 'model_tch', 'Imagecnn.json')

def get_raw_dataset(data_pars, task):
    if data_pars["dataset"] == "WIKI_QA":
        filter_train_pack_raw = data_pars["preprocess"]["train"]["filter"] \
            if "filter" in data_pars["preprocess"]["train"] else False
        filter_test_pack_raw = data_pars["preprocess"]["test"]["filter"] \
            if "filter" in data_pars["preprocess"]["test"] else False
        train_pack_raw = mz.datasets.wiki_qa.load_data('train', task=task, filtered=filter_train_pack_raw)
        test_pack_raw  = mz.datasets.wiki_qa.load_data('test', task=task, filtered=filter_test_pack_raw)
        return train_pack_raw, test_pack_raw
    else:
        dataset_name = data_pars["dataset"]
        raise Exception(f"Not support choice {dataset_name} dataset yet")

###########################################################################################################
###########################################################################################################
class Model:
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None, out_pars=None):
        ### Model Structure        ################################
        if model_pars is None :
            self.model = None
            return self
 
        _model = model_pars['model']
        assert _model in MODELS.keys()
        
        self.task = get_task(model_pars)
        
        train_pack_raw, test_pack_raw = get_raw_dataset(data_pars, self.task)
        
        _preprocessor_pars = data_pars["preprocess"]
        if "basic_preprocessor" in _preprocessor_pars:
            pars = _preprocessor_pars["basic_preprocessor"]
            preprocessor = mz.preprocessors.BasicPreprocessor(
                truncated_length_left=pars["truncated_length_left"],
                truncated_length_right=pars["truncated_length_right"],
                filter_low_freq=pars["filter_low_freq"]
            )
        else:
            preprocessor = MODELS[model_name].get_default_preprocessor()

        self.trainloader = get_data_loader(_model, preprocessor, _preprocessor_pars["train"], train_pack_raw)
        self.testloader = get_data_loader(_model, preprocessor, _preprocessor_pars["test"], test_pack_raw)

        self.model = MODELS[_model]()
        update_model_param(model_pars["params"], self.model, self.task, preprocessor)
        
        self.model.build()

def get_params(param_pars=None, **kw):
    pp          = param_pars
    choice      = pp['choice']
    config_mode = pp['config_mode']
    data_path   = pp['data_path']

    if choice == "json":
        data_path = path_norm(data_path)
        cf = json.load(open(data_path, mode='r'))
        cf = cf[config_mode]

        ####Normalize path  : add /models/dataset/
        cf['data_pars'] = path_norm_dict(cf['data_pars'])
        cf['out_pars']  = path_norm_dict(cf['out_pars'])

        return cf['model_pars'], cf['data_pars'], cf['compute_pars'], cf['out_pars']

    else:
        raise Exception(f"Not support choice {choice} yet")

def fit(model, data_pars=None, compute_pars=None, out_pars=None, **kwargs):
    model0 = model.model
    epochs = compute_pars["epochs"]

    optimize_parameters = compute_pars["optimize_parameters"] \
        if "optimize_parameters" in compute_pars else False
    if optimize_parameters:
        # Currently hardcode optimized parameters for Bert,
        # Hard to generalize.
        no_decay = ['bias', 'LayerNorm.weight']
        model_parameters = [
            {'params': [p for n, p in model0.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 5e-5},
            {'params': [p for n, p in model0.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    else:
        model_parameters = model0.parameters()
    
    optimizer_ = list(compute_pars["optimizer"].keys())[0]
    optimizer = OPTIMIZERS[optimizer_](model_parameters, compute_pars["optimizer"][optimizer_])

    trainer = mz.trainers.Trainer(
                model=model.model, 
                optimizer=optimizer, 
                trainloader=model.trainloader, 
                validloader=model.testloader,
                validate_interval=None, 
                epochs=epochs
            )
    trainer.run()
    return model, None

def predict(model, session=None, data_pars=None, compute_pars=None, out_pars=None):
    # get a batch of data
    model0 = model.model
    _, valid_iter = get_dataset(data_pars=data_pars)
    x_test        = next(iter(valid_iter))[0].to(device)
    ypred         = model0(x_test).detach().cpu().numpy()
    return ypred


def fit_metrics(model, data_pars=None, compute_pars=None, out_pars=None):
    pass


def save(model, session=None, save_pars=None):
    from mlmodels.util import save_tch
    save_tch(model=model, save_pars=save_pars)


def load(load_pars):
    from mlmodels.util import load_tch
    return load_tch(load_pars)



###########################################################################################################
###########################################################################################################
def train(data_path="dataset/", pars_choice="json", config_mode="train"):
    ### Local test

    log("#### Loading params   ##############################################")
    param_pars = {"choice":pars_choice,  "data_path":data_path,  "config_mode": config_mode}
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
    log(  data_pars, out_pars )

    log("#### Loading dataset   #############################################")
    #xtuple = get_dataset(data_pars)


    log("#### Model init, fit   #############################################")
    session = None
    model = Model(model_pars, data_pars, compute_pars)
    model, session = fit(model, data_pars, compute_pars, out_pars)


    log("#### Predict   #####################################################")
    #ypred = predict(model, session, data_pars, compute_pars, out_pars)


    log("#### metrics   #####################################################")
    #metrics_val = fit_metrics(model, data_pars, compute_pars, out_pars)
    print(metrics_val)


    log("#### Plot   ########################################################")


    log("#### Save/Load   ###################################################")
    save_pars = { "path": out_pars["path"]  }
    save(model=model, save_pars=save_pars)
    model2 = load( save_pars )
    ypred = predict(model2, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
    print(model2)


if __name__ == "__main__":
    train(data_path="model_tch/matchzoo_ranking_drmmtks.json", pars_choice="json", config_mode="train")







