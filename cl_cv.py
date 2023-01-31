from sklearn.metrics import classification_report, hamming_loss, make_scorer
from sklearn.model_selection import KFold, cross_val_predict, cross_validate
from skmultilearn.dataset import load_dataset
from utils import get_description
N_SPLITS = 5


def eval_report(classifierFactory, dataset_name):
    X_train, y_train, feature_names, label_names = load_dataset(
        set_name=dataset_name, variant="train")
    X_test, y_test, feature_names, label_names = load_dataset(
        set_name=dataset_name, variant="test")
    target_names = [label_name[0] for label_name in label_names]

    classifier = classifierFactory()

    kFold = KFold(n_splits=N_SPLITS)

    scoring = {**{item: item for item in ['accuracy',
                                          'precision_macro',  'precision_micro',
                                          'recall_macro',
                                          'recall_micro',
                                          'f1_macro',
                                          'f1_micro']}, **{'hamming': make_scorer(hamming_loss)}}

    scores = cross_validate(classifier, X_train, y_train, scoring=scoring,
                            cv=kFold, return_train_score=True, return_estimator=True)

    # Making the classification_report using the test set
    # And returning both cross validation scores and report
    # for comparison
    estimator = scores['estimator'][N_SPLITS - 1]
    y_pred = estimator.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=target_names)

    return scores, report

# This dies


def complete_report(datasets, classifiers):
    result = {}
    for datasetName in datasets:
        result[datasetName] = {}
        for classifierFactory in classifiers:
            score, report = eval_report(classifierFactory, datasetName)
            result[datasetName][get_description(
                classifierFactory)] = score, report
    return result
