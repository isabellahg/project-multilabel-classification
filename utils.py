import pandas as pd
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from skmultilearn.adapt import BRkNNaClassifier, BRkNNbClassifier, MLkNN, MLARAM, MLTSVM

DATASETS = {'scene', 'Corel5k', 'bibtex', 'enron', 'rcv1subset5', 'tmc2007_500',
            'rcv1subset3',   'rcv1subset1',   'delicious',   'rcv1subset4',   'genbase',   'birds',   'emotions',
            'rcv1subset2', 'mediamill', 'medical', 'yeast'}
TRANSFORMATION_APPROACHES = {BinaryRelevance, ClassifierChain, LabelPowerset}
ADAPTATION_APPROACHES = {BRkNNaClassifier,
                         BRkNNbClassifier,
                         MLkNN,
                         MLARAM,
                         MLTSVM}
                         
def get_description(function):
    return function.__doc__.replace('\n', '').strip()


def display_results(result, classifiers):
    for classifier in classifiers:
        name = get_description(classifier)
        score, report = result[name]
        print("## Classifier: ", name)
        print("### Cross validation training scores ")
        print(pd.DataFrame(score).transpose())
        print("### Classification report ")
        print(report)
