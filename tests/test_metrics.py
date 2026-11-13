import numpy as np
import pytest

from biomass.metrics import weighted_r2_score


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32, np.int64])
def test_integer_residuals_do_not_overflow(dtype):
    high = np.iinfo(dtype).max if dtype != np.int64 else 2**40
    truth = np.array([[0, 0], [high, high]], dtype=dtype)
    prediction = truth[::-1].copy()
    original_truth, original_prediction = truth.copy(), prediction.copy()

    score, per_target = weighted_r2_score(truth, prediction, [0.3, 0.7])

    assert score == pytest.approx(-3.0)
    np.testing.assert_allclose(per_target, [-3.0, -3.0])
    np.testing.assert_array_equal(truth, original_truth)
    np.testing.assert_array_equal(prediction, original_prediction)


def test_float_predictions_and_weighting_are_unchanged():
    truth = np.array([[0.0, 0.0], [2.0, 4.0]])
    prediction = np.array([[0.0, 4.0], [2.0, 0.0]])
    score, per_target = weighted_r2_score(truth, prediction, [3.0, 1.0])
    assert score == pytest.approx(0.0)
    np.testing.assert_allclose(per_target, [1.0, -3.0])


def test_existing_shape_validation_is_preserved():
    with pytest.raises(ValueError, match="shape"):
        weighted_r2_score(np.zeros((2, 2)), np.zeros((2, 1)), [0.5, 0.5])
