from h2o.estimators.xgboost import *
from tests import pyunit_utils



class TestGammaWrongResponseType():
    def test_response_error(self):
        assert H2OXGBoostEstimator.available()

        prostate_frame = h2o.import_file(path=pyunit_utils.locate("smalldata/prostate/prostate_complete.csv.zip"))

        x = ["ID", "AGE", "RACE", "GLEASON", "DCAPS", "PSA", "VOL", "CAPSULE"]
        y = 'DPROS'
        prostate_frame[y] = prostate_frame[y].asfactor()

        model = H2OXGBoostEstimator(training_frame=prostate_frame, learn_rate=1,
                                    booster='gbtree', distribution='gamma')


def xgboost_prostate_gamma_wrong_response_small():
    TestGammaWrongResponseType().test_response_error();


if __name__ == "__main__":
    pyunit_utils.standalone_test(xgboost_prostate_gamma_wrong_response_small)
else:
    xgboost_prostate_gamma_wrong_response_small()
