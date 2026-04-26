import pandas as pd
from src.metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


def test_attrition_rate_returns_expected_percent():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "department": ["Sales", "Sales", "HR", "HR"],
            "attrition": ["Yes", "No", "No", "Yes"],
        }
    )
    assert attrition_rate(df) == 50.0


def test_attrition_by_department_returns_expected_columns():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "department": ["Sales", "Sales", "HR", "HR"],
            "attrition": ["Yes", "No", "No", "Yes"],
        }
    )
    result = attrition_by_department(df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_rates():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "overtime": ["Yes", "Yes", "No", "No"],
            "attrition": ["Yes", "Yes", "Yes", "No"],
        }
    )
    result = attrition_by_overtime(df)
    overtime_rate = result.loc[result["overtime"] == "Yes", "attrition_rate"].iloc[0]
    no_overtime_rate = result.loc[result["overtime"] == "No", "attrition_rate"].iloc[0]
    assert overtime_rate == 100.0
    assert no_overtime_rate == 50.0


def test_average_income_by_attrition():
    df = pd.DataFrame(
        {
            "attrition": ["Yes", "Yes", "No", "No"],
            "monthly_income": [3000, 5000, 7000, 9000],
        }
    )
    result = average_income_by_attrition(df)
    assert result.loc[result["attrition"] == "Yes", "avg_monthly_income"].iloc[0] == 4000.0
    assert result.loc[result["attrition"] == "No", "avg_monthly_income"].iloc[0] == 8000.0


def test_satisfaction_summary_attrition_rate_is_within_group():
    # 4 employees with satisfaction=1, 2 left → rate should be 50%
    # 2 employees with satisfaction=4, 0 left → rate should be 0%
    # Before the fix this would divide by total leavers (2), giving 100% and 0%
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5, 6],
            "job_satisfaction": [1, 1, 1, 1, 4, 4],
            "attrition": ["Yes", "Yes", "No", "No", "No", "No"],
        }
    )
    result = satisfaction_summary(df)
    rate_low = result.loc[result["job_satisfaction"] == 1, "attrition_rate"].iloc[0]
    rate_high = result.loc[result["job_satisfaction"] == 4, "attrition_rate"].iloc[0]
    assert rate_low == 50.0
    assert rate_high == 0.0
