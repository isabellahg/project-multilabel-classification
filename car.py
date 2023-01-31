from functools import reduce
from skmultilearn.dataset import load_dataset


def instances(X):
    '''
    Number of instances (N)
    '''
    return X.shape[0]


def attributes(X):
    '''
    Number of attributes (F)
    '''
    return X.shape[1]


def labels(y):
    '''
    Number of labels (L)
    '''
    return y.shape[1]


def dl(y):
    '''
    Distinct Label Set (DL)
    '''
    # TODO
    return len(set(list(map(lambda x: sum([e*2**i for i, e in enumerate(x)]), y.toarray()))))


def pdl(y):
    '''
    Proportion of Distinct Label Set (PDL)
    '''
    # TODO
    return dl(y) / (2**y.shape[1] - 1)


def lcard(y):
    '''
    Label Cardinality (Lcard)
    '''
    # TODO
    return reduce(lambda x, y: x+y, map(len, y.data)) / y.shape[0]


def lden(y):
    '''
    Label Density LDen
    '''
    # TODO
    return lcard(y) / y.shape[1]


XMETRICS = [instances, attributes]
YMETRICS = [labels,
            dl,
            pdl,
            lcard,
            lden]


def get_description(function):
    return function.__doc__.replace('\n', '').strip()


def metrics(datasetName):
    X, y, _, _ = load_dataset(set_name=datasetName, variant="train")
    resultX = {get_description(metric): metric(X) for metric in XMETRICS}
    resultY = {get_description(metric): metric(y) for metric in YMETRICS}
    return {**resultX, **resultY}

# TODO Faltan f,g,h