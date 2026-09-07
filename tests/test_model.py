import math
import pytest
from toronto_transit_ml_platform.predict import predict_delay

@pytest.mark.requires_model
def test_real_model_prediction():
    prediction = predict_delay(
        day="Wednesday",
        line="102 MARKHAM ROAD",
        code="MFDV",
        bound="N",
        month=8,
        hour=17,
    )

    assert isinstance(prediction, float)
    assert math.isfinite(prediction)