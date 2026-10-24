"""Deprecated calibration shim.

Calibration must be measured from real model predictions and labelled data, not
fabricated from request text. Factory v4 therefore exposes no local estimator.
"""
from factory_contracts.errors import ModelUnavailableError

def evaluate_score_calibration(*_args, **_kwargs):
    raise ModelUnavailableError(
        message="Synthetic Decision calibration is disabled; use measured calibration data."
    )
