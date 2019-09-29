import pytest

from sklearn.utils.estimator_checks import check_estimator

from treatment-learn import TemplateEstimator
from treatment-learn import TemplateClassifier
from treatment-learn import TemplateTransformer


@pytest.mark.parametrize(
    "Estimator", [TemplateEstimator, TemplateTransformer, TemplateClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
