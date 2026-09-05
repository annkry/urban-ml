from urban_ml.modeling.metrics import mean_absolute_error, root_mean_squared_error


def test_mean_absolute_error_hand_computed() -> None:
    assert mean_absolute_error([1, 2, 3], [1, 2, 5]) == 2 / 3


def test_root_mean_squared_error_hand_computed() -> None:
    # errors: 0, 0, 2 -> mean squared error = 4/3 -> sqrt
    assert root_mean_squared_error([1, 2, 3], [1, 2, 5]) == (4 / 3) ** 0.5


def test_perfect_prediction_gives_zero_error() -> None:
    y = [1, 2, 3, 4]
    assert mean_absolute_error(y, y) == 0.0
    assert root_mean_squared_error(y, y) == 0.0
