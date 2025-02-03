import warnings
import math
import scipy
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.base import is_classifier
from scipy.special import xlogy


class BaseInferer():
    def __init__(
        self, 
        learner,
        remover,
        algorithm,
        *,
        loss_func = None,
        infer_type = None,
        double_split = None,
        perturb_size = None,
        n_copies = None,
        n_permutations = None,
        random_state = None,
        removed_column = None):
        if loss_func is None:
            if is_classifier(learner.estimator):
                loss_func = "log_loss"
            else:
                loss_func = "mean_squared_error"

        if loss_func == "log_loss":
            def log_loss(target, pred):
                eps = np.finfo(pred.dtype).eps
                pred = np.clip(pred, eps, 1 - eps)
                loss = -xlogy(target, pred).sum(axis=1)
                return loss
            loss_func = log_loss
            binarize = True
            response_method = "predict_proba" 

        if loss_func == "zero_one_loss":
            def zero_one_loss(target, pred):
                loss = 1 * (target == pred)
                return loss
            loss_func = zero_one_loss
            binarize = False
            response_method = "predict" 
        
        if loss_func == "mean_squared_error":
            def mean_squared_error(target, pred):
                loss = (target - pred)**2
                return loss
            loss_func = mean_squared_error
            binarize = False
            response_method = "predict"
        
        if loss_func == "mean_absolute_error":
            def mean_absolute_error(target, pred):
                loss = np.abs(target - pred)
                return loss
            loss_func = mean_absolute_error
            binarize = False
            response_method = "predict"

        n_splits = learner.n_splits
        n_repeats = learner.n_repeats
        n_folds = learner.n_folds
        
        self.learner = learner
        self.remover = remover
        self.algorithm = algorithm
        self.loss_func = loss_func
        self.infer_type = infer_type
        self.double_split = double_split
        self.perturb_size = perturb_size
        self.n_copies = n_copies
        self.n_permutations = n_permutations
        self.random_state = random_state
        self.removed_column = removed_column
        self.binarize = binarize
        self.response_method = response_method
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.n_folds = n_folds

    def summarize(
        self,
        *,
        combine = False,
        cross_fit = False,
        reverse = False,
        verbose = True
    ):
        """
        Summarize inference results
        
        Parameters
        ----------

        combine : bool, default=False
            If `True`, inference results will be combined across 
            folds and repeats. For `estimates` and `std_error`, simple 
            average is used for combination. For `p_value`, several 
            p-value combination methods are used.

        cross_fit : bool, default=False
            If `True`, inference results will be integrated across folds.

        reverse : bool, default=False
            If `True`, negative values of `estimate` will be reported. Note 
            that by default a smaller `estimate` indicates the removed 
            feature is more important.

        verbose : bool, default=True
            Controls the verbosity.

        Returns
        -------
        summary : DataFrame
            A summary for inference results.
            
        """
        n_splits = self.n_splits
        n_repeats = self.n_repeats
        n_folds = self.n_folds

        sizes, estimates, std_errors, p_values = self._stats(cross_fit)
        if reverse is True:
            estimates = - estimates
            
        if not combine or (len(p_values) == 1):
            if len(p_values) == 1:
                warnings.warn(
                    "Only one p-value is available. Combination methods are not implemented.",
                    UserWarning)
                
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
                {"size": sizes,
                 "estimate": estimates,
                 "std_error": std_errors,
                 "p_value": p_values},
                index = index)
        else:
            combined_p_values = self._combined_p_values(p_values)
            index = pd.Index(
                    combined_p_values.keys(), 
                    name = "method")
            summary = pd.DataFrame(
                {"size": [np.mean(sizes)] * len(combined_p_values),
                 "estimate": [np.mean(estimates)] * len(combined_p_values),
                 "std_error": [np.mean(std_errors)] * len(combined_p_values),
                 "p_value": combined_p_values.values()},
                index = index
            )
        if verbose:
            print("Algorithm:", self.algorithm, end = " ")
            print("(double_split=", self.double_split, ", " ,
                  "perturb_size=", self.perturb_size, ")", sep = "")        
            print("Inference Type:", self.infer_type.title(), end = " ")
            print("(n_copies=", self.n_copies, ", " ,
                  "n_permutations=", self.n_permutations, ")", sep = "")
            print("Loss Function: ", self.loss_func.__name__.replace("_", " ").title(),
                 " (", "reverse=", reverse, ")", sep = "")
        return summary

    def _stats(
        self,
        cross_fit
    ):
        l_losses = self.learner_losses_
        r_losses = self.removed_losses_
        infer_type = self.infer_type
        double_split = self.double_split
        
        n_repeats = self.n_repeats
        n_folds = self.n_folds
            
        if double_split:
            sizes = np.array([len(l_loss) + len(r_loss)
               for l_loss, r_loss in zip(l_losses, r_losses)])
        else:
            sizes = np.array([len(l_loss) for l_loss in l_losses])
        
        estimates = np.array([l_loss.mean() - r_loss.mean()
               for l_loss, r_loss in zip(l_losses, r_losses)])
        
        if (infer_type == "permutation") or (infer_type == "randomization"):
            null_values = self.null_values_
            std_errors = np.array([null_value.std() for null_value in null_values])
        else:
            if double_split:
                std_errors = np.array([np.sqrt(l_loss.var() / len(l_loss) + r_loss.var() / len(r_loss)) 
                              for l_loss, r_loss in zip(l_losses, r_losses)])
            else:
                std_errors = np.array([np.sqrt((l_loss - r_loss).var() / len(l_loss)) 
                              for l_loss, r_loss in zip(l_losses, r_losses)])
        if cross_fit:
            sizes = sizes.reshape((n_repeats, n_folds)).sum(axis = 1)
            estimates = estimates.reshape((n_repeats, n_folds)).mean(axis = 1)
            std_errors = (std_errors.reshape((n_repeats, n_folds)).mean(axis = 1) * 
                          np.sqrt(1 / n_folds))
            if (infer_type == "permutation") or (infer_type == "randomization"):
                null_values = np.array(null_values).reshape((n_repeats, n_folds, -1)).mean(axis = 1)
                p_values = np.array([(null_value < estimate).mean() 
                            for estimate, null_value in zip(estimates, null_values)])
            else:
                p_values = np.array([scipy.stats.norm.cdf(estimate / std_error) 
                            for estimate, std_error in zip(estimates, std_errors)])
        else:
            if (infer_type == "permutation") or (infer_type == "randomization"):
                p_values = np.array([(null_value < estimate).mean() 
                            for estimate, null_value in zip(estimates, null_values)])
            else:
                p_values = np.array([scipy.stats.norm.cdf(estimate / std_error) 
                            for estimate, std_error in zip(estimates, std_errors)])
        return sizes, estimates, std_errors, p_values


    def _combined_p_values(
        self,
        p_values
    ):
        const = np.sum(1 / (np.arange(len(p_values)) + 1))
        order_const = const * (len(p_values) / (np.arange(len(p_values)) + 1))
        t0 = np.mean(np.tan((.5 - np.array(p_values)) * np.pi))
        
        combined_p_values = {
            "gmean": np.e * scipy.stats.gmean(p_values, 0),
            "median": 2 * np.median(p_values, 0),
            "q1": len(p_values) / 2 * np.partition(p_values, 1)[1],
            "min": len(p_values) * np.min(p_values, 0),
            "hmean": np.e * np.log(len(p_values)) * scipy.stats.hmean(p_values, 0),
            "hommel": np.min(np.sort(p_values) * order_const),
            "cauchy": .5 - np.arctan(t0) / np.pi}
        combined_p_values = {key: np.minimum(value, 1) 
                             for key, value in combined_p_values.items()}
        return combined_p_values

    
    def _removed_column(
        self,
        learner,
        remover
    ):
        cols = np.arange(learner.X_.shape[1])
        if hasattr(learner.X_, "iloc") and hasattr(remover.y_, "iloc"):
            X, y = learner.X_.values, remover.y_.values
        else:
            X, y = learner.X_, remover.y_
        for col in cols:
            tester = np.array_equal(y, X[:,col])
            if tester:
                removed_column = col
                break
        return removed_column



class CIT(BaseInferer):
    def infer(
        self
    ):
        learner = self.learner
        remover = self.remover
        algorithm = self.algorithm
        loss_func = self.loss_func
        infer_type = self.infer_type
        n_copies = self.n_copies
        n_permutations = self.n_permutations
        random_state = self.random_state
        removed_column = self.removed_column
        binarize = self.binarize
        response_method = self.response_method
        
        l_features = learner._features()
        l_targets = learner._targets(binarize)
        l_preds = learner._preds(response_method)
        l_losses = [loss_func(l_target, l_pred)
                   for l_target, l_pred in zip(l_targets, l_preds)]
        
        r_features = l_features
        r_rvs = remover._rvs(
            n_copies = n_copies,
            random_state = random_state)
        r_losses = []

        def _r_loss_i(r_rv_i):
            if hasattr(r_feature, "iloc"):
                r_feature.iloc[:, removed_column] = r_rv_i
            else:
                r_feature[:, removed_column] = r_rv_i
            r_pred_i = learner.predict(
                r_feature, 
                split = split)
            r_loss_i = loss_func(l_target, r_pred_i)
            return r_loss_i
        
        for split, (l_loss, l_target, r_feature, r_rv) in enumerate(
            zip(l_losses, l_targets, r_features, r_rvs)):
            r_loss = np.apply_along_axis(
                _r_loss_i,
                axis = 1,
                arr = r_rv)
            r_losses.append(r_loss)
        
        if infer_type == "randomization":
            null_values = []  
            for split, (l_loss, r_loss) in enumerate(
                zip(l_losses, r_losses)):
                null_value = r_loss.mean(axis = 1) - r_loss.mean() 
                null_values.append(null_value)
                r_loss = r_loss.mean(axis = 0)
                r_losses[split] = r_loss
        else:
            for split, (l_loss, r_loss) in enumerate(
                zip(l_losses, r_losses)):
                r_loss = r_loss.mean(axis = 0)
                r_losses[split] = r_loss
            
            if infer_type == "permutation":
                null_values = [] 
                rng = np.random.default_rng(random_state)
                for l_loss, r_loss in zip(l_losses, r_losses):
                    paired_loss = np.column_stack([l_loss, r_loss])
                    null_value = []
                    for permutation in range(n_permutations):
                        permuted_loss = rng.permuted(
                            paired_loss, 
                            axis = 1)
                        null_value.append(
                            permuted_loss[:,0].mean() - 
                            permuted_loss[:,1].mean())
                    null_value = np.array(null_value)
                    null_values.append(null_value)
            else:
                null_values = None

        self.learner_losses_ = l_losses
        self.removed_losses_ = r_losses
        self.null_values_ = null_values


class RIT(BaseInferer):
    def infer(
        self
    ):
        learner = self.learner
        remover = self.remover
        algorithm = self.algorithm
        loss_func = self.loss_func
        infer_type = self.infer_type
        n_permutations = self.n_permutations
        double_split = self.double_split
        perturb_size = self.perturb_size
        random_state = self.random_state
        binarize = self.binarize
        response_method = self.response_method

        l_targets = learner._targets(binarize)
        l_preds = learner._preds(response_method)
        l_losses = [loss_func(l_target, l_pred)
                   for l_target, l_pred in zip(l_targets, l_preds)]
        
        r_targets = remover._targets(binarize)
        r_preds = remover._preds(response_method)
        r_losses = [loss_func(r_target, r_pred)
                   for r_target, r_pred in zip(r_targets, r_preds)]
        if double_split:
            l_losses = [l_loss[[i for i in range(len(l_loss)) if i % 2 == 0]] 
                        for l_loss in l_losses]
            r_losses = [r_loss[[i for i in range(len(r_loss)) if i % 2 == 1]] 
                        for r_loss in r_losses]

        if perturb_size is not None:
            rng = np.random.default_rng(random_state)
            r_losses = [r_loss + rng.normal(
                scale = perturb_size, 
                size = len(r_loss)) for r_loss in r_losses]
            
        if infer_type == "permutation":
            null_values = [] 
            rng = np.random.default_rng(random_state)
            if double_split:
                for l_loss, r_loss in zip(l_losses, r_losses):
                    concated_loss = np.concatenate([l_loss, r_loss])
                    size = len(concated_loss)
                    null_value = []
                    for permutation in range(n_permutations):
                        permuted_loss = rng.permuted(concated_loss)
                        null_value.append(
                            permuted_loss[:math.ceil(size/2)].mean() - 
                            permuted_loss[math.ceil(size/2):].mean())
                    null_value = np.array(null_value)
                    null_values.append(null_value)                
            else:
                for l_loss, r_loss in zip(l_losses, r_losses):
                    paired_loss = np.column_stack([l_loss, r_loss])
                    null_value = []
                    for permutation in range(n_permutations):
                        permuted_loss = rng.permuted(
                            paired_loss, 
                            axis = 1)
                        null_value.append(
                            permuted_loss[:,0].mean() - 
                            permuted_loss[:,1].mean())
                    null_value = np.array(null_value)
                    null_values.append(null_value)
        else:
            null_values = None
        
        self.learner_losses_ = l_losses
        self.removed_losses_ = r_losses
        self.null_values_ = null_values

class Inferer(BaseInferer):
    """
    An inferer for making statistical inference

    Parameters
    ----------
    learner : crosser object 
        The learner being infered. 

    remover : crosser object
        A remover for removing the effect of $x_j$. Note that 
        its cross-validator (splitter) must be identical to the 
        learner.
        
            - When `algorithm` is "CRT", "HRT", "RPT", 
              or "CPI", the remover should be a crosser that 
              predicts $x_j$ by $x_{-j}$.
            - When `algorithm` is "LOCO", "BBT", or "PIE", 
              the remover should be a crosser that predicts 
              $y$ by $x_{-j}$.
    
    algorithm : {"CRT", "HRT", "RPT", "CPI", "LOCO", "BBT", "PIE"}
        The algorithm for inference.

    loss_func : {"mean_squared_error", "mean_absolute_error", \
                "zero_one_loss", "log_loss"}, default=None
        The loss function for measuring the difference between 
        $y$ and its prediction. 

            - When `learner` is a regressor, `None` means 
              "mean_squared_error".
            - When `learner` is a classifier, `None` means 
              "log_loss".

    infer_type : {"randomization", "normality", "permutation"}, \
                 default=None
        The method for constructing the reference distribution of 
        feature importance.
        
            - When `algorithm` is "CRT", "HRT", "RPT", 
              `infer_type` must be "randomization" (default).
            - When `algorithm` is "CPI", "LOCO", "BBT", or "PIE", 
              `infer_type` can be "normality" (default) or 
              "permutation".     

    double_split : bool, default=None
        Whether double split is used if `algorithm` is "LOCO", 
        "BBT", or "PIE". If applicable, `None` means `True`.

    perturb_size : non-negative float, default=None
        The standard deviation of random error for pertubation 
        when `algorithm` is "LOCO", "BBT", or "PIE". If applicable, 
        `None` means 0.

    n_copies : int, default=None
        The number of copies for sampling $x_j$ given $x_{-j}$ 
        when `algorithm` is "CRT", "HRT", "RPT", or "CPI".
        
            - When `algorithm` is "CRT", "HRT", "RPT", `None` means
              2000.
            - When `algorithm` is "CPI", `None` means 1.
              
    n_permutations : int, default=None
        The number of permutations if `infer_type` is "permutation".
        If applicable, `None` means 2000 (default).

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of sampling and permutation.

    removed_column : int, default=None
        The column number of $x_j$. It is automatically detected 
        by defaulty.
    
    """
    def __init__(
        self,
        learner,
        remover,
        algorithm,
        *,
        loss_func = None,
        infer_type = None,
        n_copies = None,
        double_split = None,
        perturb_size = None,
        n_permutations = None,
        random_state = None,
        removed_column = None
    ):
        if algorithm in ["CRT", "HRT","RPT", "CPI"]:
            if algorithm  in ["CRT", "HRT","RPT"]:
                if infer_type is None:
                    infer_type = "randomization"
                if n_copies is None:
                    n_copies = 2000
            else:
                if infer_type is None:
                    infer_type = "normality"
                if n_copies is None:
                    n_copies = 1
                if (infer_type == "permutation") and (n_permutations is None):
                    n_permutations = 2000
            if removed_column is None:
                removed_column = self._removed_column(learner, remover)
            self.__class__ = CIT
            CIT.__init__(
                self,
                learner, 
                remover, 
                algorithm,
                loss_func = loss_func,
                infer_type = infer_type,
                n_copies = n_copies,
                n_permutations = n_permutations,
                random_state = random_state,
                removed_column = removed_column)
        elif algorithm in ["LOCO", "BBT", "PIE"]:
            if infer_type is None:
                infer_type = "normality"
            if (infer_type == "permutation") and (n_permutations is None):
                n_permutations = 2000
            if algorithm in ["BBT", "PIE"] and (double_split is None):
                double_split = True
            if algorithm in ["LOCO"] and (loss_func is None):
                loss_func = "mean_absolute_error"
            self.__class__ = RIT
            RIT.__init__(
                self,
                learner, 
                remover, 
                algorithm,
                loss_func = loss_func,
                infer_type = infer_type,
                double_split = double_split,
                perturb_size = perturb_size,
                n_permutations = n_permutations,
                random_state = random_state)