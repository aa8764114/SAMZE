from abc import ABCMeta, abstractmethod

import numpy as np

from scipy.special import xlogy

from ._base import BaseEnsemble
from ..base import ClassifierMixin, RegressorMixin, is_classifier, is_regressor

from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils import check_array, check_random_state, _safe_indexing
from ..utils.extmath import softmax
from ..utils.extmath import stable_cumsum
from ..metrics import accuracy_score, r2_score
from ..utils.validation import check_is_fitted
from ..utils.validation import _check_sample_weight
from ..utils.validation import has_fit_parameter
from ..utils.validation import _num_samples
from ..utils.validation import _deprecate_positional_args

#123test
#2021/10/22 push test
#----------------------------------------------------------------
"""自己import的東西"""


from .data_preprocessing_pipeline import *
from .feature_engineering import *
from sklearn.pipeline import Pipeline
from scipy.special import softmax


#----------------------------------------------------------------



__all__ = [
    'AdaBoostClassifier',
    'AdaBoostClassifierZe',
    'AdaBoostRegressor',
]


class BaseWeightBoosting(BaseEnsemble, metaclass=ABCMeta):
    """Base class for AdaBoost estimators.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator=None, *,
                 n_estimators=50,
                 estimator_params=tuple(),
                 learning_rate=1.,
                 random_state=None):

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        self.learning_rate = learning_rate
        self.random_state = random_state

    def _check_X(self, X):
        return check_array(X, accept_sparse=['csr', 'csc'], ensure_2d=True,
                           allow_nd=True, dtype=None)

    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier/regressor from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        Returns
        -------
        self : object
        """
        # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        X, y = self._validate_data(X, y,
                                   accept_sparse=['csr', 'csc'],
                                   ensure_2d=True,
                                   allow_nd=True,
                                   dtype=None,
                                   y_numeric=is_regressor(self))

        sample_weight = _check_sample_weight(sample_weight, X, np.float64)
        sample_weight /= sample_weight.sum()
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative weights")

        # Check parameters
        self._validate_estimator()

        # Clear any previous fit results
        self.estimators_ = []
        #self.estimators_ = str(self.estimators_)
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)
        
        #建模前挑選的特徵
        self.estimators_features_ = []
        

        # Initializion of the random number instance that will be used to
        # generate a seed at each iteration
        random_state = check_random_state(self.random_state)

        for iboost in range(self.n_estimators):
            # Boosting step
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight,
                random_state)

            # Early termination
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            #print(estimator_weight)
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break
            '''
            #如果挑不到特徵就停
            #print('挑到的特徵數:', len(self.estimators_features_))
            X_temp = pd.DataFrame(X)            
            #print('輸入資料欄位數量:', len(X_temp.columns))
            # 如果挑不到特徵就停
            if len(self.estimators_features_) <= len(X_temp.columns):
                break
            '''
            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum

        return self

    @abstractmethod
    def _boost(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost.

        Warning: This method needs to be overridden by subclasses.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        sample_weight : array-like of shape (n_samples,)
            The current sample weights.

        random_state : RandomState
            The current random number generator

        Returns
        -------
        sample_weight : array-like of shape (n_samples,) or None
            The reweighted sample weights.
            If None then boosting has terminated early.

        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.

        error : float
            The classification error for the current boost.
            If None then boosting has terminated early.
        """
        pass

    def staged_score(self, X, y, sample_weight=None):
        """Return staged scores for X, y.

        This generator method yields the ensemble score after each iteration of
        boosting and therefore allows monitoring, such as to determine the
        score on a test set after each boost.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            Labels for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Yields
        ------
        z : float
        """
        X = self._check_X(X)

        for y_pred in self.staged_predict(X):
            if is_classifier(self):
                yield accuracy_score(y, y_pred, sample_weight=sample_weight)
            else:
                yield r2_score(y, y_pred, sample_weight=sample_weight)

    @property
    def feature_importances_(self):
        """The impurity-based feature importances.

        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            The feature importances.
        """
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise ValueError("Estimator not fitted, "
                             "call `fit` before `feature_importances_`.")

        try:
            norm = self.estimator_weights_.sum()
            return (sum(weight * clf.feature_importances_ for weight, clf
                    in zip(self.estimator_weights_, self.estimators_))
                    / norm)

        except AttributeError as e:
            raise AttributeError(
                "Unable to compute feature importances "
                "since base_estimator does not have a "
                "feature_importances_ attribute") from e

    

def _samme_proba(estimator, n_classes, X):
    """Calculate algorithm 4, step 2, equation c) of Zhu et al [1].

    References
    ----------
    .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

    """
    #print(X.shape[1])
    proba = estimator.predict_proba(X)

    # Displace zero probabilities so the log is defined.
    # Also fix negative elements which may occur with
    # negative sample weights.
    np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)
    log_proba = np.log(proba)

    return (n_classes - 1) * (log_proba - (1. / n_classes)
                              * log_proba.sum(axis=1)[:, np.newaxis])



#輸入弱分類器預測機率，實際答案，輸出弱分類器錯誤率(自己的方法)
#error: all,看所有資料計算錯誤率  semi,只看分錯的資料計算錯誤率
def samze_weight(proba, act_ans, error):
    
    #print(proba)
    #print(type(proba))
    #print(proba.shape)
    
    
    #將預測機率四捨五入
    proba_temp = []

    for i in proba:
        #print(i)
        #print(type(i))
        sumone= softmax(i)#將數值總和調整為1
        zun = list(np.around(sumone, 5))#把數值四捨五入以免存不下
        #print(zun)
        proba_temp.append(zun)    
    proba = proba_temp
    
    #存下預測答案index
    predit_ans = []
    for i in proba:
        predit_ans.append(i.index(max(i)))   
    
    #將預測錯誤資料 wrong_prob/sum_prob
    wrong_prob = [] #錯的部份的機率和
    sum_prob = [] #全部機率相加
    
    
    for p, pa, aa in zip(proba, predit_ans, act_ans):
        sum_prob.append(sum(p))
        
        if(error == 'semi'):
            if(pa != aa):
                wrong_prob.append(sum(p)-i[aa])
        else:
            wrong_prob.append(sum(p)-i[aa])

    #print(sum(wrong_prob))
    #print(sum(sum_prob))
    return(sum(wrong_prob)/sum(sum_prob))


#將每筆資料預測與實際答案對照，輸出每筆資料在預測中錯誤的機率
def error_prob(proba, act_ans):
    
    #print(proba)
    #print(type(proba))
    #print(proba.shape)
    
    
    #將預測機率四捨五入
    proba_temp = []

    for i in proba:
        #print(i)
        #print(type(i))
        sumone= softmax(i)#將數值總和調整為1
        zun = list(np.around(sumone, 5))#把數值四捨五入以免存不下
        #print(zun)
        proba_temp.append(zun)    
    proba = proba_temp
    
    #存下預測答案index
    predit_ans = []
    for i in proba:
        predit_ans.append(i.index(max(i)))   
    
    #將預測錯誤資料 wrong_prob/sum_prob
    wrong_prob = [] #錯的部份的機率和
    sum_prob = [] #全部機率相加
    
    
    for p, pa, aa in zip(proba, predit_ans, act_ans):
        sum_prob.append(sum(p))
        wrong_prob.append(sum(p)-i[aa])

    return  wrong_prob



#實際答案在預測結果的預測機率
#proba:弱分類器預測機率
#act_ans:實際答案
def act_ans_prob(proba, act_ans):

    #將預測機率四捨五入
    proba_temp = []
    for i in proba:
        #print(i)
        #print(type(i))
        sumone= softmax(i)#將數值總和調整為1
        zun = list(np.around(sumone, 5))#把數值四捨五入以免存不下
        #print(zun)
        proba_temp.append(zun)    
    proba = proba_temp
    
    #存下實際答案在預測結果的預測機率
    actans_prob = []
    for aa, p in zip(act_ans, proba):
        actans_prob.append(p[aa])
    
    return actans_prob

#輸入決策樹模型，輸出屬性重要性排序（index）
#參數：決策數模型，想要前幾%重要的屬性
def importance_to_index(clf, prob):
    importances = list(clf.feature_importances_)
    s = importances
    s2 = sorted(range(len(s)), key=lambda k: s[k], reverse=True)
    return s2[:int(len(s2)*prob)]

#將屬性index編號轉換為屬性名字
def index_to_colname(indexs, oridata):
    colname = []
    for i in indexs:
        colname.append(list(x_.columns)[i])
    
    return(colname)
    
class AdaBoostClassifier(ClassifierMixin, BaseWeightBoosting):

    def __init__(self,
                 base_estimator=None, *,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

        self.algorithm = algorithm

    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, the sample weights are initialized to
            ``1 / n_samples``.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Check that algorithm is supported
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Fit
        return super().fit(X, y, sample_weight)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super()._validate_estimator(
            default=DecisionTreeClassifier(max_depth=1))

        #  SAMME-R requires predict_proba-enabled base estimators
        if self.algorithm == 'SAMME.R':
            if not hasattr(self.base_estimator_, 'predict_proba'):
                raise TypeError(
                    "AdaBoostClassifier with algorithm='SAMME.R' requires "
                    "that the weak learner supports the calculation of class "
                    "probabilities with a predict_proba method.\n"
                    "Please change the base estimator or set "
                    "algorithm='SAMME' instead.")
        if not has_fit_parameter(self.base_estimator_, "sample_weight"):
            raise ValueError("%s doesn't support sample_weight."
                             % self.base_estimator_.__class__.__name__)

    def _boost(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost.

        Perform a single boost according to the real multi-class SAMME.R
        algorithm or to the discrete SAMME algorithm and return the updated
        sample weights.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        sample_weight : array-like of shape (n_samples,)
            The current sample weights.

        random_state : RandomState instance
            The RandomState instance used if the base estimator accepts a
            `random_state` attribute.

        Returns
        -------
        sample_weight : array-like of shape (n_samples,) or None
            The reweighted sample weights.
            If None then boosting has terminated early.

        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.

        estimator_error : float
            The classification error for the current boost.
            If None then boosting has terminated early.
        """
        if self.algorithm == 'SAMME.R':
            return self._boost_real(iboost, X, y, sample_weight, random_state)

        else:  # elif self.algorithm == "SAMME":
            return self._boost_discrete(iboost, X, y, sample_weight,
                                        random_state)

    def _boost_real(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME.R real algorithm."""
        estimator = self._make_estimator(random_state=random_state)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict_proba = estimator.predict_proba(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1),
                                       axis=0)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        # Construct y coding as described in Zhu et al [2]:
        #
        #    y_k = 1 if c == k else -1 / (K - 1)
        #
        # where K == n_classes_ and c, k in [0, K) are indices along the second
        # axis of the y coding with c being the index corresponding to the true
        # class label.
        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1. / (n_classes - 1), 1.])
        y_coding = y_codes.take(classes == y[:, np.newaxis])

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba = y_predict_proba  # alias for readability
        np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)

        # Boost weight using multi-class AdaBoost SAMME.R alg
        estimator_weight = (-1. * self.learning_rate
                            * ((n_classes - 1.) / n_classes)
                            * xlogy(y_coding, y_predict_proba).sum(axis=1))

        # Only boost the weights if it will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight *
                                    ((sample_weight > 0) |
                                     (estimator_weight < 0)))

        return sample_weight, 1., estimator_error

    def _boost_discrete(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME discrete algorithm."""
        estimator = self._make_estimator(random_state=random_state)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict = estimator.predict(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        # Instances incorrectly classified
        incorrect = y_predict != y
        #print('預測錯誤的:', incorrect)


#---------------------------------------------------------------------------------------------------------        
        # Error fraction(SAMME弱分類器錯誤率)
        #'''
        #原方法
        estimator_error_o = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))
        #print('SAMME弱分類器錯誤率 : ', estimator_error_o)
        #'''
        
        #'''
        #我的方法
        n_classes = self.n_classes_        
        proba = _samme_proba(estimator, n_classes, X)
        #print('SAMME弱分類器預測機率:', proba)
        
        act_ans = list(y)

        #print('轉list:', list(act_ans))
        #print('實際答案:', act_ans)
        #print('答案維度:', act_ans.shape)
        #print('答案型態:', type(act_ans))

        #estimator_error_my_all = samze_weight(proba, act_ans, error = 'all') #不管分對或錯都計算錯誤率
        #estimator_error_my_semi = samze_weight(proba, act_ans, error = 'semi') #只有分錯的計算錯誤率       
        #print('SAMME改弱分類器錯誤率 : ', estimator_error_my)
        #'''

        #方法一，原方法        
        estimator_error = estimator_error_o
        
        #方法二，自己方法，不管分對或錯都計算錯誤率        
        #estimator_error = estimator_error_my_all

        #方法三，自己方法，只有分錯的計算錯誤率        
        #estimator_error = estimator_error_my_semi
        
        #方法四，相加除二(全部)
        #estimator_error = (estimator_error_o + estimator_error_my_all) / 2
        
        #方法五，相加除二(只有分錯)
        #estimator_error = (estimator_error_o + estimator_error_my_semi) / 2        
        
        #方法六，相乘開根號(全部)
        #estimator_error = (estimator_error_o * estimator_error_my_all) ** 0.5

        #方法七，相乘開根號(只有分錯)
        #estimator_error = (estimator_error_o * estimator_error_my_semi) ** 0.5
        
        #方法五，將 各資料的權重*實際答案在弱分類器的預測機率 平均
        error_probs = error_prob(proba, act_ans)
        #print('實際答案在弱分類器的預測機率 : ', len(actans_prob))
        estimator_error = np.mean(np.average(error_probs, weights=sample_weight, axis=0))
        #print('SAMME弱分類器錯誤率 : ', estimator_error)

        
        if(estimator_error == 0.5):
            estimator_error = 0.50000000000001
        elif(estimator_error == 1.0):
            estimator_error = 0.99999999999999
        print('弱分類器錯誤率:', estimator_error)

#---------------------------------------------------------------------------------------------------------  

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_
        
        #弱分類器錯誤率太高停止
        '''
        # Stop if the error is at least as bad as random guessing
        if estimator_error >= 1. - (1. / n_classes):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError('BaseClassifier in AdaBoostClassifier '
                                 'ensemble is worse than random, ensemble '
                                 'can not be fit.')
            return None, None, None
        '''
        
        # Boost weight using multi-class AdaBoost SAMME alg
        estimator_weight = self.learning_rate * (
            np.log((1. - estimator_error) / estimator_error) +
            np.log(n_classes - 1.))
        
        print('SAMME弱分類器權重:', estimator_weight, '\n')

        # Only boost the weights if I will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight * incorrect *
                                    (sample_weight > 0))

        return sample_weight, estimator_weight, estimator_error

    def predict(self, X):
        """Predict classes for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        X = self._check_X(X)

        pred = self.decision_function(X)

        if self.n_classes_ == 2:
            return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def staged_predict(self, X):
        """Return staged predictions for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        This generator method yields the ensemble prediction after each
        iteration of boosting and therefore allows monitoring, such as to
        determine the prediction on a test set after each boost.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Yields
        ------
        y : generator of ndarray of shape (n_samples,)
            The predicted classes.
        """
        X = self._check_X(X)

        n_classes = self.n_classes_
        classes = self.classes_

        if n_classes == 2:
            for pred in self.staged_decision_function(X):
                yield np.array(classes.take(pred > 0, axis=0))

        else:
            for pred in self.staged_decision_function(X):
                yield np.array(classes.take(
                    np.argmax(pred, axis=1), axis=0))

    def decision_function(self, X):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        score : ndarray of shape of (n_samples, k)
            The decision function of the input samples. The order of
            outputs is the same of that of the :term:`classes_` attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        check_is_fitted(self)
        X = self._check_X(X)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]

        if self.algorithm == 'SAMME.R':
            # The weights are all 1. for SAMME.R
            pred = sum(_samme_proba(estimator, n_classes, X)
                       for estimator in self.estimators_)
        else:  # self.algorithm == "SAMME"
            pred = sum((estimator.predict(X) == classes).T * w
                       for estimator, w in zip(self.estimators_,
                                               self.estimator_weights_))

        pred /= self.estimator_weights_.sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            return pred.sum(axis=1)
        return pred

    def staged_decision_function(self, X):
        """Compute decision function of ``X`` for each boosting iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each boosting iteration.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Yields
        ------
        score : generator of ndarray of shape (n_samples, k)
            The decision function of the input samples. The order of
            outputs is the same of that of the :term:`classes_` attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        check_is_fitted(self)
        X = self._check_X(X)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = None
        norm = 0.

        for weight, estimator in zip(self.estimator_weights_,
                                     self.estimators_):
            norm += weight

            if self.algorithm == 'SAMME.R':
                # The weights are all 1. for SAMME.R
                current_pred = _samme_proba(estimator, n_classes, X)
            else:  # elif self.algorithm == "SAMME":
                current_pred = estimator.predict(X)
                current_pred = (current_pred == classes).T * weight

            if pred is None:
                pred = current_pred
            else:
                pred += current_pred

            if n_classes == 2:
                tmp_pred = np.copy(pred)
                tmp_pred[:, 0] *= -1
                yield (tmp_pred / norm).sum(axis=1)
            else:
                yield pred / norm

    @staticmethod
    def _compute_proba_from_decision(decision, n_classes):
        """Compute probabilities from the decision function.

        This is based eq. (4) of [1] where:
            p(y=c|X) = exp((1 / K-1) f_c(X)) / sum_k(exp((1 / K-1) f_k(X)))
                     = softmax((1 / K-1) * f(X))

        References
        ----------
        .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost",
               2009.
        """
        if n_classes == 2:
            decision = np.vstack([-decision, decision]).T / 2
        else:
            decision /= (n_classes - 1)
        return softmax(decision, copy=False)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities of the classifiers
        in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of
            outputs is the same of that of the :term:`classes_` attribute.
        """
        check_is_fitted(self)
        X = self._check_X(X)

        n_classes = self.n_classes_

        if n_classes == 1:
            return np.ones((_num_samples(X), 1))

        decision = self.decision_function(X)
        return self._compute_proba_from_decision(decision, n_classes)

    def staged_predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities of the classifiers
        in the ensemble.

        This generator method yields the ensemble predicted class probabilities
        after each iteration of boosting and therefore allows monitoring, such
        as to determine the predicted class probabilities on a test set after
        each boost.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Yields
        -------
        p : generator of ndarray of shape (n_samples,)
            The class probabilities of the input samples. The order of
            outputs is the same of that of the :term:`classes_` attribute.
        """
        X = self._check_X(X)

        n_classes = self.n_classes_

        for decision in self.staged_decision_function(X):
            yield self._compute_proba_from_decision(decision, n_classes)

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the weighted mean predicted class log-probabilities of the classifiers
        in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of
            outputs is the same of that of the :term:`classes_` attribute.
        """
        X = self._check_X(X)
        return np.log(self.predict_proba(X))


#Adaboost加入特徵選擇
class AdaBoostClassifierZe(ClassifierMixin, BaseWeightBoosting):

    def __init__(self,
                 base_estimator=None, *,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None,
                 
                 #我自己加的參數
                 fs_enable='anova_kf',
                 estimator_error_calc=0,

                 ):

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

        self.algorithm = algorithm
        
        self.fs_enable = fs_enable
        self.estimator_error_calc = estimator_error_calc

    def fit(self, X, y, sample_weight=None):

        # Check that algorithm is supported
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Fit
        return super().fit(X, y, sample_weight)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super()._validate_estimator(
            default=DecisionTreeClassifier(max_depth=1))

        #  SAMME-R requires predict_proba-enabled base estimators
        if self.algorithm == 'SAMME.R':
            if not hasattr(self.base_estimator_, 'predict_proba'):
                raise TypeError(
                    "AdaBoostClassifier with algorithm='SAMME.R' requires "
                    "that the weak learner supports the calculation of class "
                    "probabilities with a predict_proba method.\n"
                    "Please change the base estimator or set "
                    "algorithm='SAMME' instead.")
        if not has_fit_parameter(self.base_estimator_, "sample_weight"):
            raise ValueError("%s doesn't support sample_weight."
                             % self.base_estimator_.__class__.__name__)

    def _boost(self, iboost, X, y, sample_weight, random_state):

        if self.algorithm == 'SAMME.R':
            return self._boost_real(iboost, X, y, sample_weight, random_state)

        else:  # elif self.algorithm == "SAMME":
            return self._boost_discrete(iboost, X, y, sample_weight,
                                        random_state)

    def _boost_real(self, iboost, X, y, sample_weight, random_state):

#-------------------------------------------------------------------------------
        
        """轉dataframe再處理"""
        yo = y #抄一份原來的y
        
        X = pd.DataFrame(X)
        XO = X #抄一份原來的X(因為之後要套用選擇的特徵變dataframe才能處理)        
        y = pd.DataFrame(y)

        y.columns = ['y']
        
        df_merge = X.merge(y, how='inner', left_index=True, right_index=True)#輸出入資料合併
        #print(df_merge)        
        
        """自己做有放回抽樣"""
        #print(sample_weight)
        df_merge = df_merge.sample(frac=1, replace=1, weights=sample_weight, axis='index')

        """把輸出入屬性拆開"""
        #取出y的行名
        y_name = df_merge.columns[-1]
        
        #輸入屬性
        X = df_merge.drop([y_name], axis=1)#同上方法刪除行

        #目標屬性
        y = df_merge[y_name]


        """自己做特徵選擇"""
        if(self.fs_enable == 'anova_kf'):
            #X = pd.DataFrame(X)
            #print(X[:10])
            
            #95%信心水準挑選
            pipe_fs = Pipeline(
                   [
                     ('filter', feature_selection(method = 'anova_kf', p_threshold = 0.05))
                   ])

            X1 = pipe_fs.fit_transform(X, y)
         
            #50%信心水準挑選
            pipe_fs = Pipeline(
                   [
                     ('filter', feature_selection(method = 'anova_kf', p_threshold = 0.5))
                   ])

            X2 = pipe_fs.fit_transform(X, y)        
            
            #不挑
            X3 = X
            
            #條件判斷
            if(len(X1.columns)!=0): #95%有特
                X = X1
                print('X1')

            elif(len(X1.columns)==0):#95%沒特，看50%有無

                if(len(X2.columns)!=0): #50%有特
                    X = X2
                    print('X2')
                    
                elif(len(X2.columns)==0):
                    X = X3
                    print('X3')

            print('弱分類器輸入資料維度 : ', len(X.columns))           

            self.estimators_features_.append(X.columns)#把該弱分類器挑選的屬性存下來
            XO = XO[X.columns]#把屬性選擇後的屬性套用在原始資料
            
            X = X.to_numpy()
            y = y.to_numpy()

            
            #print(y)
            #print("-----------------------")
            #print(yo)

        elif(self.fs_enable == 'gini'):

            clf = DecisionTreeClassifier(criterion=self.fs_enable, random_state=0)
            clf.fit(X, y)
            importance = importance_to_index(clf, 0.5)

            X = X[importance]

            print('重要屬性index:', importance)




        else:
            X = X.to_numpy()
            y = y.to_numpy()            
#-------------------------------------------------------------------------------

        """Implement a single boost using the SAMME.R real algorithm."""
        estimator = self._make_estimator(random_state=random_state)
        
        
        #estimator.fit(X, y, sample_weight=sample_weight)
        #把抽樣拿到外面做
        estimator.fit(X, y)
        
        #y_predict_proba = estimator.predict_proba(X)
        #因為抽樣過資料動過了所以把原資料拿來用(要調整維度)
        y_predict_proba = estimator.predict_proba(XO)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1),
                                       axis=0)

        # Instances incorrectly classified
        #print('臥草 : ', len(yo))
        #incorrect = y_predict != y
        incorrect = y_predict != yo #應該要跟所有資料比

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))
        print("SAMME.R弱分類器錯誤率:", estimator_error)

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

#------------------------------------------------------------------------------            
        #自己加入的
        # Stop if the error is at least as bad as random guessing
        #如果弱分類器太爛就停
        n_classes = self.n_classes_
        '''
        if estimator_error >= 1. - (1. / n_classes):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError('BaseClassifier in AdaBoostClassifier '
                                 'ensemble is worse than random, ensemble '
                                 'can not be fit.')
            return None, None, None
        '''
        # Boost weight using multi-class AdaBoost SAMME alg
        #SAMME弱分類器權重
        samme_estimator_weight = self.learning_rate * (
            np.log((1. - estimator_error) / estimator_error) +
            np.log(n_classes - 1.))
        print("SAMME.R弱分類器權重:", samme_estimator_weight)
#------------------------------------------------------------------------------  
        # Construct y coding as described in Zhu et al [2]:
        #
        #    y_k = 1 if c == k else -1 / (K - 1)
        #
        # where K == n_classes_ and c, k in [0, K) are indices along the second
        # axis of the y coding with c being the index corresponding to the true
        # class label.
        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1. / (n_classes - 1), 1.])
        
        #應該是要把所有資料編碼
        #y_coding = y_codes.take(classes == y[:, np.newaxis])        
        y_coding = y_codes.take(classes == yo[:, np.newaxis])

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba = y_predict_proba  # alias for readability
        np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)

        # Boost weight using multi-class AdaBoost SAMME.R alg
        #SAMME.R資料權重
        estimator_weight = (-1. * self.learning_rate
                            * ((n_classes - 1.) / n_classes)
                            * xlogy(y_coding, y_predict_proba).sum(axis=1))

        # Only boost the weights if it will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight *
                                    ((sample_weight > 0) |
                                     (estimator_weight < 0)))

        #原來的回傳值
        #return sample_weight, 1., estimator_error
        #把自己做的弱分類器權重回傳
        return sample_weight, samme_estimator_weight, estimator_error
        
    def _boost_discrete(self, iboost, X, y, sample_weight, random_state):
    
#-------------------------------------------------------------------------------
        """轉dataframe再處理"""
        yo = y #抄一份原來的y
        
        X = pd.DataFrame(X)
        XO = X #抄一份原來的X(因為之後要套用選擇的特徵變dataframe才能處理)        
        y = pd.DataFrame(y)

        y.columns = ['y']
        
        df_merge = X.merge(y, how='inner', left_index=True, right_index=True)#輸出入資料合併
        #print(df_merge)        
        
        """自己做有放回抽樣"""
        #print(sample_weight)
        df_merge = df_merge.sample(frac=1, replace=1, weights=sample_weight, axis='index')

        """把輸出入屬性拆開"""
        #取出y的行名
        y_name = df_merge.columns[-1]
        
        #輸入屬性
        X = df_merge.drop([y_name], axis=1)#同上方法刪除行

        #目標屬性
        y = df_merge[y_name]

        """自己做特徵選擇"""
        if(self.fs_enable == True):

            #X = pd.DataFrame(X)
            #print(X[:10])
            
            #95%信心水準挑選
            pipe_fs = Pipeline(
                   [
                     ('filter', feature_selection(method = 'anova_kf', p_threshold = 0.05))
                   ])

            X1 = pipe_fs.fit_transform(X, y)
         
            #50%信心水準挑選
            pipe_fs = Pipeline(
                   [
                     ('filter', feature_selection(method = 'anova_kf', p_threshold = 0.5))
                   ])

            X2 = pipe_fs.fit_transform(X, y)        
            
            #不挑
            X3 = X
            
            #條件判斷
            if(len(X1.columns)!=0): #95%有特
                X = X1
                print('X1')

            elif(len(X1.columns)==0):#95%沒特，看50%有無

                if(len(X2.columns)!=0): #50%有特
                    X = X2
                    print('X2')
                    
                elif(len(X2.columns)==0):
                    X = X3
                    print('X3')

            print('弱分類器輸入資料維度 : ', len(X.columns))           

            self.estimators_features_.append(X.columns)#把該弱分類器挑選的屬性存下來
            XO = XO[X.columns]#把屬性選擇後的屬性套用在原始資料
            
            X = X.to_numpy()
            y = y.to_numpy()

            
            #print(y)
            #print("-----------------------")
            #print(yo)
        
        else:
            self.estimators_features_ = X.columns
            X = X.to_numpy()
            y = y.to_numpy()  
#-------------------------------------------------------------------------------    
    
    
        """Implement a single boost using the SAMME discrete algorithm."""
        estimator = self._make_estimator(random_state=random_state)

        #自己做抽樣，不在建模時做
        #estimator.fit(X, y, sample_weight=sample_weight)
        estimator.fit(X, y)
        
        #用所有資料評估
        #y_predict = estimator.predict(X)        
        y_predict = estimator.predict(XO)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        # Instances incorrectly classified
        #用原始的y對答案
        #incorrect = y_predict != y
        incorrect = y_predict != yo
        
#---------------------------------------------------------------------------------------------------------        
        # Error fraction(SAMME弱分類器錯誤率)
        #'''
        #原方法
        estimator_error_o = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))
        #print('SAMME弱分類器錯誤率 : ', estimator_error_o)
        #'''
        
        #'''
        #我的方法
        n_classes = self.n_classes_        
        proba = _samme_proba(estimator, n_classes, X)
        #print('SAMME弱分類器預測機率:', proba)
        
        act_ans = list(y)

        #print('轉list:', list(act_ans))
        #print('實際答案:', act_ans)
        #print('答案維度:', act_ans.shape)
        #print('答案型態:', type(act_ans))

        
               
        #print('SAMME改弱分類器錯誤率 : ', estimator_error_my)
        #'''

        #方法零，原方法
        if(self.estimator_error_calc == 0):
            estimator_error = estimator_error_o
        
        #方法一，自己方法，不管分對或錯都計算錯誤率
        elif(self.estimator_error_calc == 1):
            estimator_error_my_all = samze_weight(proba, act_ans, error = 'all') #不管分對或錯都計算錯誤率
            estimator_error = estimator_error_my_all

        #方法二，自己方法，只有分錯的計算錯誤率
        elif(self.estimator_error_calc == 2): 
            estimator_error_my_semi = samze_weight(proba, act_ans, error = 'semi') #只有分錯的計算錯誤率
            estimator_error = estimator_error_my_semi
        
        #方法三，相加除二(全部)
        elif(self.estimator_error_calc == 3):
            estimator_error_my_all = samze_weight(proba, act_ans, error = 'all') #不管分對或錯都計算錯誤率
            estimator_error = (estimator_error_o + estimator_error_my_all) / 2
        
        #方法四，相加除二(只有分錯)
        elif(self.estimator_error_calc == 4):
            estimator_error_my_semi = samze_weight(proba, act_ans, error = 'semi') #只有分錯的計算錯誤率
            estimator_error = (estimator_error_o + estimator_error_my_semi) / 2        
        
        #方法五，相乘開根號(全部)
        elif(self.estimator_error_calc == 5):
            estimator_error_my_all = samze_weight(proba, act_ans, error = 'all') #不管分對或錯都計算錯誤率
            estimator_error = (estimator_error_o * estimator_error_my_all) ** 0.5

        #方法六，相乘開根號(只有分錯)
        elif(self.estimator_error_calc == 6):
            estimator_error_my_semi = samze_weight(proba, act_ans, error = 'semi') #只有分錯的計算錯誤率
            estimator_error = (estimator_error_o * estimator_error_my_semi) ** 0.5
        
        #方法七，將 各資料的權重*實際答案在弱分類器的預測機率 平均
        elif(self.estimator_error_calc == 7):          
            error_probs = error_prob(proba, act_ans)
            #print('實際答案在弱分類器的預測機率 : ', len(actans_prob))
            estimator_error = np.mean(np.average(error_probs, weights=sample_weight, axis=0))
            #print('SAMME弱分類器錯誤率 : ', estimator_error)
        else:
            estimator_error = estimator_error_o
        
        if(estimator_error == 0.5):
            estimator_error = 0.50000000000001
        elif(estimator_error == 1.0):
            estimator_error = 0.99999999999999
        print('弱分類器錯誤率:', estimator_error)

#---------------------------------------------------------------------------------------------------------  


        # Stop if the error is at least as bad as random guessing
        '''
        if estimator_error >= 1. - (1. / n_classes):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError('BaseClassifier in AdaBoostClassifier '
                                 'ensemble is worse than random, ensemble '
                                 'can not be fit.')
            return None, None, None
        '''

        # Boost weight using multi-class AdaBoost SAMME alg
        #SAMME弱分類器權重
        estimator_weight = self.learning_rate * (
            np.log((1. - estimator_error) / estimator_error) +
            np.log(n_classes - 1.))
        print('SAMME弱分類器權重:', estimator_weight)

        # Only boost the weights if I will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight * incorrect *
                                    (sample_weight > 0))

        return sample_weight, estimator_weight, estimator_error

    def predict(self, X):

        X = self._check_X(X)

        pred = self.decision_function(X)

        if self.n_classes_ == 2:
            return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def staged_predict(self, X):

        X = self._check_X(X)

        n_classes = self.n_classes_
        classes = self.classes_

        if n_classes == 2:
            for pred in self.staged_decision_function(X):
                yield np.array(classes.take(pred > 0, axis=0))

        else:
            for pred in self.staged_decision_function(X):
                yield np.array(classes.take(
                    np.argmax(pred, axis=1), axis=0))

    def decision_function(self, X):
        X = pd.DataFrame(X)
        #print('臥草:', type(X))
        #print('NMSL:', X)
        #print(self.estimators_features_)
        #print(type(self.estimators_features_[0][0]))
        #for fsl in self.estimators_features_:
            #print(list(fsl))
            #print(X[list(fsl)].shape[1])
        
        #print(self.estimators_)
        #print('estimator_weights_ : ', len(self.estimator_weights_))
        #print(self.estimator_weights_)

        check_is_fitted(self)
        X = self._check_X(X)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
#-------------------------------------------------------------------------------         
        '''
        #原程式
        if self.algorithm == 'SAMME.R':
            # The weights are all 1. for SAMME.R
            pred = sum(_samme_proba(estimator, n_classes, X)
                       for estimator in self.estimators_)
        else:  # self.algorithm == "SAMME"
            pred = sum((estimator.predict(X) == classes).T * w
                       for estimator, w in zip(self.estimators_,
                                               self.estimator_weights_))
        '''
#-------------------------------------------------------------------------------
        
        #自己改的2(最後用這版)
        X = pd.DataFrame(X)#要改成dataframe才可以挑屬性        
        if self.algorithm == 'SAMME.R':
            # The weights are all 1. for SAMME.R
            pred = sum(_samme_proba(estimator, n_classes, X[list(f)])
                       for estimator, f in zip(self.estimators_, self.estimators_features_))
        else:  # self.algorithm == "SAMME"
            pred = sum((estimator.predict(X[list(f)]) == classes).T * w
                       for estimator, w, f in zip(self.estimators_,
                                               self.estimator_weights_, self.estimators_features_))
        X = X.to_numpy()#做完再改回ndarray
        pred = np.array(pred)
        
        #print(X.shape)
        #print(type(X))
        #print(X)
        #print(pred)
        #print(type(pred))
        #print(len(pred))        
#-------------------------------------------------------------------------------
                                               

        pred /= self.estimator_weights_.sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            return pred.sum(axis=1)
        return pred

#------------------------------------------------------------------------------- 
    def staged_decision_function(self, X):

        check_is_fitted(self)
        X = self._check_X(X)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = None
        norm = 0.

        for weight, estimator in zip(self.estimator_weights_,
                                     self.estimators_):
            norm += weight

            if self.algorithm == 'SAMME.R':
                # The weights are all 1. for SAMME.R
                current_pred = _samme_proba(estimator, n_classes, X)
            else:  # elif self.algorithm == "SAMME":
                current_pred = estimator.predict(X)
                current_pred = (current_pred == classes).T * weight

            if pred is None:
                pred = current_pred
            else:
                pred += current_pred

            if n_classes == 2:
                tmp_pred = np.copy(pred)
                tmp_pred[:, 0] *= -1
                yield (tmp_pred / norm).sum(axis=1)
            else:
                yield pred / norm

    @staticmethod
    def _compute_proba_from_decision(decision, n_classes):

        if n_classes == 2:
            decision = np.vstack([-decision, decision]).T / 2
        else:
            decision /= (n_classes - 1)
        return softmax(decision, copy=False)

    def predict_proba(self, X):

        check_is_fitted(self)
        X = self._check_X(X)

        n_classes = self.n_classes_

        if n_classes == 1:
            return np.ones((_num_samples(X), 1))

        decision = self.decision_function(X)
        return self._compute_proba_from_decision(decision, n_classes)

    def staged_predict_proba(self, X):

        X = self._check_X(X)

        n_classes = self.n_classes_

        for decision in self.staged_decision_function(X):
            yield self._compute_proba_from_decision(decision, n_classes)

    def predict_log_proba(self, X):

        X = self._check_X(X)
        return np.log(self.predict_proba(X))
        
    #輸入特徵回傳權重
    def get_feature_weight(self,feature, fs):#想要找的特徵，弱分類器特徵+權重
        
        weight_sum = 0
        for i in fs:
            if((feature in i[1]) == True):
                weight_sum += i[0]
        
        return weight_sum    

    #特徵選擇
    def feature_selection(self, percent, feature_and_weight = None):
        
        estimator_weights_ = self.estimator_weights_
        estimator_features_ = self.estimators_features_
        
        #把弱分類器權重及挑到特徵合併 [權重, 特徵]
        fs = []
        for w, f in zip(estimator_weights_, estimator_features_):
            temp = []
            temp.append(w)
            temp.append(f)

            fs.append(temp)    

        #所有弱分類器有挑到的屬性
        all_features = []
        for i in estimator_features_:
            i = list(i)
            #print(i)
            all_features += i

        select_feature_ = set(all_features)
        select_feature_ = list(select_feature_)

        #存下特徵及對應權重
        ans = []
        for i in select_feature_:
            fwt = []
            fwt.append(i)
            #print(i)
            fwt.append(self.get_feature_weight(i, fs))
            #print(get_feature_weight(i, fs))
            ans.append(fwt)

        #將結果排序
        ans = sorted(ans, key = lambda x:x[1], reverse=True)
        #輸出特徵比例
        ans = ans[:int(len(ans)*percent)]
        
        #看要不要輸出權重
        if (feature_and_weight == True):
            return(ans)
        else:
            ans2 = []
            for i in ans:
                ans2.append(i[0])
            return(ans2)


class AdaBoostRegressor(RegressorMixin, BaseWeightBoosting):
    """An AdaBoost regressor.

    An AdaBoost [1] regressor is a meta-estimator that begins by fitting a
    regressor on the original dataset and then fits additional copies of the
    regressor on the same dataset but where the weights of instances are
    adjusted according to the error of the current prediction. As such,
    subsequent regressors focus more on difficult cases.

    This class implements the algorithm known as AdaBoost.R2 [2].

    Read more in the :ref:`User Guide <adaboost>`.

    .. versionadded:: 0.14

    Parameters
    ----------
    base_estimator : object, default=None
        The base estimator from which the boosted ensemble is built.
        If ``None``, then the base estimator is
        :class:`~sklearn.tree.DecisionTreeRegressor` initialized with
        `max_depth=3`.

    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.

    learning_rate : float, default=1.
        Weight applied to each classifier at each boosting iteration. A higher
        learning rate increases the contribution of each classifier. There is
        a trade-off between the `learning_rate` and `n_estimators` parameters.

    loss : {'linear', 'square', 'exponential'}, default='linear'
        The loss function to use when updating the weights after each
        boosting iteration.

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given at each `base_estimator` at each
        boosting iteration.
        Thus, it is only used when `base_estimator` exposes a `random_state`.
        In addition, it controls the bootstrap of the weights used to train the
        `base_estimator` at each boosting iteration.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    estimator_weights_ : ndarray of floats
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : ndarray of floats
        Regression error for each estimator in the boosted ensemble.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances if supported by the
        ``base_estimator`` (when based on decision trees).

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    Examples
    --------
    >>> from sklearn.ensemble import AdaBoostRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_features=4, n_informative=2,
    ...                        random_state=0, shuffle=False)
    >>> regr = AdaBoostRegressor(random_state=0, n_estimators=100)
    >>> regr.fit(X, y)
    AdaBoostRegressor(n_estimators=100, random_state=0)
    >>> regr.predict([[0, 0, 0, 0]])
    array([4.7972...])
    >>> regr.score(X, y)
    0.9771...

    See Also
    --------
    AdaBoostClassifier, GradientBoostingRegressor,
    sklearn.tree.DecisionTreeRegressor

    References
    ----------
    .. [1] Y. Freund, R. Schapire, "A Decision-Theoretic Generalization of
           on-Line Learning and an Application to Boosting", 1995.

    .. [2] H. Drucker, "Improving Regressors using Boosting Techniques", 1997.

    """
    @_deprecate_positional_args
    def __init__(self,
                 base_estimator=None, *,
                 n_estimators=50,
                 learning_rate=1.,
                 loss='linear',
                 random_state=None):

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

        self.loss = loss
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        """Build a boosted regressor from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values (real numbers).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        Returns
        -------
        self : object
        """
        # Check loss
        if self.loss not in ('linear', 'square', 'exponential'):
            raise ValueError(
                "loss must be 'linear', 'square', or 'exponential'")

        # Fit
        return super().fit(X, y, sample_weight)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super()._validate_estimator(
            default=DecisionTreeRegressor(max_depth=3))

    def _boost(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost for regression

        Perform a single boost according to the AdaBoost.R2 algorithm and
        return the updated sample weights.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,)
            The current sample weights.

        random_state : RandomState
            The RandomState instance used if the base estimator accepts a
            `random_state` attribute.
            Controls also the bootstrap of the weights used to train the weak
            learner.
            replacement.

        Returns
        -------
        sample_weight : array-like of shape (n_samples,) or None
            The reweighted sample weights.
            If None then boosting has terminated early.

        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.

        estimator_error : float
            The regression error for the current boost.
            If None then boosting has terminated early.
        """
        estimator = self._make_estimator(random_state=random_state)

        # Weighted sampling of the training set with replacement
        bootstrap_idx = random_state.choice(
            np.arange(_num_samples(X)), size=_num_samples(X), replace=True,
            p=sample_weight
        )

        # Fit on the bootstrapped sample and obtain a prediction
        # for all samples in the training set
        X_ = _safe_indexing(X, bootstrap_idx)
        y_ = _safe_indexing(y, bootstrap_idx)
        estimator.fit(X_, y_)
        y_predict = estimator.predict(X)

        error_vect = np.abs(y_predict - y)
        sample_mask = sample_weight > 0
        masked_sample_weight = sample_weight[sample_mask]
        masked_error_vector = error_vect[sample_mask]

        error_max = masked_error_vector.max()
        if error_max != 0:
            masked_error_vector /= error_max

        if self.loss == 'square':
            masked_error_vector **= 2
        elif self.loss == 'exponential':
            masked_error_vector = 1. - np.exp(-masked_error_vector)

        # Calculate the average loss
        estimator_error = (masked_sample_weight * masked_error_vector).sum()

        if estimator_error <= 0:
            # Stop if fit is perfect
            return sample_weight, 1., 0.

        elif estimator_error >= 0.5:
            # Discard current estimator only if it isn't the only one
            if len(self.estimators_) > 1:
                self.estimators_.pop(-1)
            return None, None, None

        beta = estimator_error / (1. - estimator_error)

        # Boost weight using AdaBoost.R2 alg
        estimator_weight = self.learning_rate * np.log(1. / beta)

        if not iboost == self.n_estimators - 1:
            sample_weight[sample_mask] *= np.power(
                beta, (1. - masked_error_vector) * self.learning_rate
            )

        return sample_weight, estimator_weight, estimator_error

    def _get_median_predict(self, X, limit):
        # Evaluate predictions of all estimators
        predictions = np.array([
            est.predict(X) for est in self.estimators_[:limit]]).T

        # Sort the predictions
        sorted_idx = np.argsort(predictions, axis=1)

        # Find index of median prediction for each sample
        weight_cdf = stable_cumsum(self.estimator_weights_[sorted_idx], axis=1)
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        median_idx = median_or_above.argmax(axis=1)

        median_estimators = sorted_idx[np.arange(_num_samples(X)), median_idx]

        # Return median predictions
        return predictions[np.arange(_num_samples(X)), median_estimators]

    def predict(self, X):
        """Predict regression value for X.

        The predicted regression value of an input sample is computed
        as the weighted median prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted regression values.
        """
        check_is_fitted(self)
        X = self._check_X(X)

        return self._get_median_predict(X, len(self.estimators_))

    def staged_predict(self, X):
        """Return staged predictions for X.

        The predicted regression value of an input sample is computed
        as the weighted median prediction of the classifiers in the ensemble.

        This generator method yields the ensemble prediction after each
        iteration of boosting and therefore allows monitoring, such as to
        determine the prediction on a test set after each boost.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        Yields
        -------
        y : generator of ndarray of shape (n_samples,)
            The predicted regression values.
        """
        check_is_fitted(self)
        X = self._check_X(X)

        for i, _ in enumerate(self.estimators_, 1):
            yield self._get_median_predict(X, limit=i)
