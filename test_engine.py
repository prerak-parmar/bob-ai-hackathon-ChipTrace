"""
test_engine.py
Basic unit tests for engine.py — run with: pytest test_engine.py -v
(or just `python test_engine.py` for a quick manual run without pytest)
"""

import pandas as pd
from engine import (
    load_and_validate,
    compute_summary,
    compute_feature_importance,
    train_risk_model,
    predict_new_lot,
    generate_recommendations,
)


def make_sample_df():
    return pd.read_csv("wafer_data.csv")


def test_load_and_validate_clean_data():
    df = make_sample_df()
    cleaned, warnings = load_and_validate(df)
    assert len(warnings) == 0
    assert len(cleaned) == len(df)
    assert "Lot" in cleaned.columns


def test_load_and_validate_missing_column():
    df = make_sample_df().drop(columns=["Pressure"])
    cleaned, warnings = load_and_validate(df)
    assert any("Missing required column" in w for w in warnings)


def test_load_and_validate_drops_bad_rows():
    df = make_sample_df()
    df["Yield"] = df["Yield"].astype(object)
    df.loc[0, "Yield"] = "not_a_number"
    cleaned, warnings = load_and_validate(df)
    assert len(cleaned) == len(df) - 1
    assert any("Dropped" in w for w in warnings)


def test_summary_keys_present():
    df, _ = load_and_validate(make_sample_df())
    summary = compute_summary(df)
    for key in ["total_lots", "average_yield", "low_yield_lots", "worst_lot"]:
        assert key in summary


def test_feature_importance_sums_to_100():
    df, _ = load_and_validate(make_sample_df())
    imp_df, _ = compute_feature_importance(df)
    assert abs(imp_df["Importance"].sum() - 100) < 0.01
    assert len(imp_df) > 0


def test_predict_risky_lot_flags_abnormal_params():
    df, _ = load_and_validate(make_sample_df())
    trained = train_risk_model(df)
    result = predict_new_lot(trained, {
        "Temperature": 475, "Pressure": 2.9, "Power": 850,
        "Defects": 20, "GasFlow": 120,
    })
    assert result["risk"] in ("LOW", "MEDIUM", "HIGH")
    assert result["risk"] == "HIGH"  # this combination should clearly be high risk
    flagged_factors = [f["factor"] for f in result["findings"]]
    assert "Defects" in flagged_factors  # 20 defects is well above healthy median


def test_predict_healthy_lot_is_low_risk():
    df, _ = load_and_validate(make_sample_df())
    trained = train_risk_model(df)
    result = predict_new_lot(trained, {
        "Temperature": 450, "Pressure": 2.2, "Power": 800,
        "Defects": 2, "GasFlow": 120,
    })
    assert result["risk"] == "LOW"
    assert result["predicted_yield"] > 95


def test_predict_missing_input_raises():
    df, _ = load_and_validate(make_sample_df())
    trained = train_risk_model(df)
    try:
        predict_new_lot(trained, {"Temperature": 450})  # missing required fields
        assert False, "Expected ValueError for missing inputs"
    except ValueError:
        pass


def test_compute_spc_limits():
    df, _ = load_and_validate(make_sample_df())
    from engine import compute_spc_limits
    spc = compute_spc_limits(df["Yield"])
    assert "mean" in spc and "ucl" in spc and "lcl" in spc
    assert spc["ucl"] >= spc["mean"] >= spc["lcl"]


def test_generate_wafer_map():
    from engine import generate_wafer_map
    wafer = generate_wafer_map("L001", defect_count=12, yield_pct=88.5)
    assert len(wafer) > 100
    assert "status" in wafer.columns
    assert set(wafer["status"].unique()).issubset({"Pass", "Defect"})


def test_batch_predict_lots():
    df, _ = load_and_validate(make_sample_df())
    trained = train_risk_model(df)
    from engine import batch_predict_lots
    sample_lots = df.head(5)[trained["features"] + ["Lot"]]
    results = batch_predict_lots(trained, sample_lots)
    assert len(results) == 5
    assert "Predicted_Yield" in results.columns
    assert "Risk_Level" in results.columns


if __name__ == "__main__":
    # Manual runner if pytest isn't installed
    import sys
    tests = [
        test_load_and_validate_clean_data,
        test_load_and_validate_missing_column,
        test_load_and_validate_drops_bad_rows,
        test_summary_keys_present,
        test_feature_importance_sums_to_100,
        test_predict_risky_lot_flags_abnormal_params,
        test_predict_healthy_lot_is_low_risk,
        test_predict_missing_input_raises,
        test_compute_spc_limits,
        test_generate_wafer_map,
        test_batch_predict_lots,
    ]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {t.__name__} -> {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)

