import math
import scipy
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn import model_selection
from sklearn.base import is_classifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import get_scorer, get_scorer_names


class Crosser():
    """
    A crosser for repeated cross-fitting

    Parameters
    ----------
    estimator : estimator object
        The base estimator for repeated cross-fitting. 

    cv : cross-validation generator
        The cross-validation generator for repeated cross-fitting. 
        It must be a class member of `KFold`, `RepeatedKFold`, 
        `StratifiedKFold`, `RepeatedStratifiedKFold`, `ShuffleSplit`, 
        `StratifiedShuffleSplit`.
        
    scoring : str or callable, default=None
        Strategy to evaluate the performance of the cross-validated 
        model when using `summarize`. To see availble scoring methods 
        via `str`, use `sklearn.get_scorer_names()`. 
        
            - For regression tasks, `None` means "r2". 
            - For classification tasks, `None` means "accuracy".

    """

    def __init__(
        self,
        estimator,
        cv,
        scoring = None
    ):
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        
        self._check()
        self._setup()

    def fit(
        self, 
        X, 
        y
    ):
        """
        Fit by repeated cross-fitting
        
        Parameters
        ----------

        X : array-like of shape (n_cases, n_features)
            Training array, where `n_cases` is the number of cases and
            `n_features` is the number of features.

        y : array-like of shape (n_cases,)
            Target relative to X for classification or regression.

        Returns
        -------
        self : object
            Instance of fitted crosser.
        """
        cv = self.cv
        cv_indexes = []
        estimators = []
        for train_index, test_index in cv.split(X, y):     
            estimator = clone(self.estimator)
            if hasattr(X, "iloc"):
                X_train = X.iloc[train_index,]
            else:
                X_train = X[train_index,]
            if hasattr(y, "iloc"):
                y_train = y.iloc[train_index]
            else:
                y_train = y[train_index]
            _ = estimator.fit(X_train, y_train)
            estimators.append(estimator)               
            cv_indexes.append((train_index, test_index))
        self.X_, self.y_ = X.copy(), y.copy()
        self.estimators_ = estimators
        self.cv_indexes_ = cv_indexes
        if is_classifier(self.estimator):
            label_binarizer = LabelBinarizer()
            _ = label_binarizer.fit(y)
            self.label_binarizer_ = label_binarizer
        return self

    
    def summarize(
        self,
        *,
        combine = False,
        cross_fit = False,
        reverse = False,
        verbose = True
    ):
        """
        Summarize repeated cross-fitting results
        
        Parameters
        ----------

        combine : bool, default=False
            If `True`, score values will be combined across folds 
            and repeats.

        cross_fit : bool, default=False
            If `True`, score values will be averaged across folds.

        reverse : bool, default=False
            If `True`, negative score values will be reported. 
            Note that by default a larger score value means better 
            in `scikit-learn`.
        
        verbose : bool, default=True
            Controls the verbosity.

        Returns
        -------
        summary : DataFrame
            A summary for validation, train, and test scores (`val_score`, 
            `train_score`, and `test_score`). The validation scores 
            are only presented if `estimator.best_score` is available. 
            
        """
        n_splits = self.n_splits
        n_repeats = self.n_repeats
        n_folds = self.n_folds
        
        val_scores, train_scores, test_scores = self._scores(cross_fit)
        if reverse is True:
            val_scores, train_scores, test_scores = -val_scores, -train_scores, -test_scores
            
        if not combine:
            if cross_fit:
                index = pd.Index(
                    range(n_repeats), 
                    name = "repeat")
            else:
                splits = list(range(self.n_splits))
                repeats = [split // n_folds for split in splits]
                folds = [split % n_folds for split in splits]
                index = pd.MultiIndex.from_tuples(
                    list(zip(splits, repeats, folds)),
                    names = ["split", "repeat", "fold"])
            summary = pd.DataFrame(
                {"val_score": val_scores,
                 "train_score": train_scores,
                 "test_score": test_scores},
                index = index
            )
        else:
            index = pd.Index(
                    ["mean", "std"], 
                    name = "stat")
            summary = pd.DataFrame(
                {"val_score": [np.mean(val_scores), np.std(val_scores)],
                 "train_score": [np.mean(train_scores), np.std(train_scores)],
                 "test_score": [np.mean(test_scores), np.std(test_scores)]},
                index = index
            ) 
        if verbose:        
            print("Estimator: ", 
                  self.estimator.__class__.__name__, sep = "")
            print("Cross-Validator: ", 
                  self.cv.__class__.__name__, 
                  " (", "n_repeats=", self.n_repeats, 
                  ", n_folds=", self.n_folds, ")", sep = "")
            print("Scoring Function: ", 
                  self.scoring.replace("_", " ").title(), 
                  " (", "reverse=", reverse, ")", sep = "")
        return summary
        
    def predict(
        self, 
        X,
        *,
        split = None
    ):
        if split is None:
            preds = np.array(
                [estimator.predict(X) 
                 for estimator in self.estimators_])
            if is_classifier(self.estimator):
                pred = np.apply_along_axis(
                    lambda x: np.argmax(np.bincount(x)),
                    axis = 0,
                    arr = preds)
            else:
                pred = preds.mean(axis = 0)
        else:
            pred = self.estimators_[split].predict(X)
        return pred

    
    def predict_proba(
        self, 
        X,
        *,
        split = None
    ):
        if split is None:
            preds = np.array(
                [estimator.predict_proba(X)
                 for estimator in self.estimators_])
            pred = preds.mean(axis = 0)
        else:
            pred = self.estimators_[split].predict_proba(X)
        return pred

    
    def predict_log_proba(
        self, 
        X,
        *,
        split = None
    ):
        if split is None:
            preds = np.array(
                [estimator.predict_log_proba(X)
                 for estimator in self.estimators_])
            pred = preds.mean(axis = 0)
        else:
            pred = self.estimators_[split].predict_log_proba(X)
        return pred
            
    def decision_function(
        self, 
        X,
        *,
        split = None
    ):
        if split is None:
            preds = np.array(
                [estimator.decision_function(X) 
                 for estimator in self.estimators_])
            pred = preds.mean(axis = 0)
        else:
            pred = self.estimators_[split].decision_function(X)
        return pred
            
    
    def sample(
        self,
        X,
        *,
        split = None,
        n_copies = None, 
        random_state = None
    ):
        rng = np.random.default_rng(random_state)
        if is_classifier(self.estimator):
            pred = self.predict_proba(
                X, 
                split = split)
            if n_copies is None:            
                rv = rng.multinomial(1, pred)
                rv = self.label_binarizer_.inverse_transform(rv)
            else:
                rv = rng.multinomial(
                    1, pred, (n_copies, len(pred)))
                rv = np.array(
                    [self.label_binarizer_.inverse_transform(rv_i) 
                     for rv_i in rv])
        else:
            if split is None:
                targets = self._targets()
                preds = self._preds()
                residual = np.concatenate(
                    [target - pred 
                     for target, pred in zip(targets, preds)])
            else:
                feature = self._feature(split)
                target = self._target(split)
                pred = self.predict(
                    feature, 
                    split = split)     
                residual = target - pred
            pred = self.predict(
                X, 
                split = split)
            if len(pred) > len(residual):
                replace = True
            else:
                replace = False
            if n_copies is None:
                rv = pred + rng.choice(residual, len(pred), replace)
            else:
                rv = pred + np.array(
                    [rng.choice(residual, len(pred), replace) 
                     for repeat in range(n_copies)])
        return rv


    def _scores(
        self,
        cross_fit
    ):
        X = self.X_
        y = self.y_
        cv_indexes = self.cv_indexes_ 
        estimators = self.estimators_
        scorer = self.scorer
        n_repeats = self.n_repeats
        n_folds = self.n_folds
        
        train_scores = []
        val_scores = []
        test_scores = []
        for split, estimator in enumerate(estimators):
            X_train = self._feature(split, "train")
            X_test = self._feature(split, "test")
            y_train = self._target(split, False, "train")
            y_test = self._target(split, False, "test")    
            
            train_score = scorer(estimator, X_train, y_train)
            test_score = scorer(estimator, X_test, y_test)
            if hasattr(estimator, "best_score_"):
                val_score = estimator.best_score_
            else:
                val_score = np.nan        
            val_scores.append(val_score)
            train_scores.append(train_score)
            test_scores.append(test_score)
        val_scores = np.array(val_scores)
        train_scores = np.array(train_scores)
        test_scores = np.array(test_scores)
        if cross_fit:
            val_scores = val_scores.reshape((n_repeats, n_folds)).mean(axis = 1)
            train_scores = train_scores.reshape((n_repeats, n_folds)).mean(axis = 1)
            test_scores = test_scores.reshape((n_repeats, n_folds)).mean(axis = 1)
        return val_scores, train_scores, test_scores
                   
    
    def _feature(
        self,
        split,
        on = "test"
    ):
        X = self.X_
        train_index, test_index = self.cv_indexes_[split]
        if on == "test":
            index = test_index
        else:
            index = train_index
        if hasattr(X, "iloc"):
            feature = X.iloc[index, :]
        else:
            feature = X[index, :]
        return feature

    
    def _features(
        self,
        on = "test"
    ):
        n_splits = self.n_splits
        features = [self._feature(split, on) 
                    for split in range(n_splits)]
        return features

    
    def _target(
        self,
        split,
        binarize = None,
        on = "test"
    ):
        y = self.y_
        train_index, test_index = self.cv_indexes_[split]
        if on == "test":
            index = test_index
        else:
            index = train_index
        if hasattr(y, "iloc"):
            target = y.iloc[index]
        else:
            target = y[index]
        if is_classifier(self.estimator):
            if binarize is True:
                target = self.label_binarizer_.transform(target)
                if target.shape[1] == 1:
                    target = np.append(1 - target, target, axis=1)
        return target

    
    def _targets(
        self,
        binarize = None,
        on = "test"
    ):
        n_splits = self.n_splits
        targets = [self._target(split, binarize, on) 
                   for split in range(n_splits)]
        return targets

    
    def _pred(
        self,
        split,
        response_method = "predict"
    ):
        predict_func = getattr(self, response_method)
        feature = self._feature(split)
        pred = predict_func(feature, split = split)
        return pred

    
    def _preds(
        self,
        response_method = "predict"
    ):
        n_splits = self.n_splits
        preds = [self._pred(split, response_method) 
                 for split in range(n_splits)]
        return preds

    
    def _rvs(
        self,
        n_copies = None,
        random_state = None
    ):
        n_splits = self.n_splits
        rvs = []
        for split in range(n_splits):
            feature = self._feature(split)
            rv = self.sample(
                feature, 
                split = split, 
                n_copies = n_copies,
                random_state = random_state)
            rvs.append(rv)
        return rvs

    def _check(
        self
    ):
        estimator = self.estimator
        cv = self.cv
        scoring = self.scoring
        
        allowed_cvs = {"KFold", "StratifiedKFold", 
                  "RepeatedKFold", "RepeatedStratifiedKFold",
                  "ShuffleSplit", "StratifiedShuffleSplit"}
        if not isinstance(
            self.cv,
            tuple(getattr(model_selection, allowed_cv) 
                  for allowed_cv in allowed_cvs)
        ):
            raise ValueError("Support `cv` types are {}.".format(allowed_cvs))
        
        if scoring is not None:
            if not (isinstance(scoring, str) or callable(scoring)):
                raise ValueError("Support `scoring` types are `str`, `callable`, or `None`")
            if isinstance(scoring, str):
                scorer_names = get_scorer_names()
                if not scoring in get_scorer_names():
                    raise ValueError("Support `scoring` names are {}.".format(scorer_names))


    def _setup(
        self
    ):
        estimator = self.estimator
        cv = self.cv
        scoring = self.scoring
        
        kf_cvs = {"KFold", "StratifiedKFold", 
                  "RepeatedKFold", "RepeatedStratifiedKFold"}
        ss_cvs = {"ShuffleSplit", "StratifiedShuffleSplit"}
        n_splits = cv.get_n_splits()
        if isinstance(
            cv, 
            tuple(getattr(model_selection, kf_cv) 
                  for kf_cv in kf_cvs)):
            if hasattr(cv, "n_repeats"):
                n_repeats = cv.n_repeats
            else:
                n_repeats = 1
        else:
            n_repeats = cv.get_n_splits()
        n_folds = n_splits // n_repeats

        if scoring is None:
            if is_classifier(estimator):
                scoring = "accuracy"
            else:
                scoring = "r2"
        if isinstance(scoring, str):
            scorer = get_scorer(scoring)
        elif callable(scoring):
            scorer = scoring
            scoring = "callable"
                    
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.n_folds = n_folds
        self.scorer = scorer
        



