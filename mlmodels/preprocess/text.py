import torchtext

def default_tokenizer(text):
    import spacy
    spacy_en = spacy.load('en')
    return [token.text for token in spacy_en.tokenizer(text)]

def embeddingLoader(embedding_path, train=True, download=True, transform=None, model_pars={}, data_pars={}):
    d = model_pars

    # text_field = data.Field(sequential=True,
    #                         tokenize=tokenizer,
    #                         include_lengths=False,
    #                         use_vocab=True)
    vec = torchtext.vocab.Vectors(
        name=d.get('embedding_name'), cache=d.get('embedding_path'), url=d.get('embedding_url', None))
    # text_field.build_vocab()
    print(vec.__dict__.keys())
    print(type(vec['vectors']))
    print(vec['stoi'])
    exit()