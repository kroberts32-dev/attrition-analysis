import pandas as pd
import pytest
from src.metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


@pytest.fixture
def dept_df():
    """Sales: 2 of 3 left (66.67%). HR: 1 of 3 left (33.33%)."""
    return pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5, 6],
            "department": ["Sales", "Sales", "Sales", "HR", "HR", "HR"],
            "overtime": ["Yes", "Yes", "No", "No", "No", "No"],
            "monthly_income": [3000, 5000, 7000, 4000, 6000, 8000],
            "job_satisfaction": [1, 2, 3, 2, 3, 4],
            "attrition": ["Yes", "Yes", "No", "Yes", "No", "No"],
        }
    )


# --- attrition_rate ---

def test_attrition_rate_fifty_percent():
    df = pd.DataFrame({"employee_id": [1, 2, 3, 4], "attrition": ["Yes", "No", "No", "Yes"]})
    assert attrition_rate(df) == 50.0


def test_attrition_rate_zero():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["No", "No"]})
    assert attrition_rate(df) == 0.0


def test_attrition_rate_one_hundred():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["Yes", "Yes"]})
    assert attrition_rate(df) == 100.0


def test_attrition_rate_rounds_to_two_decimals():
    # 1 of 3 = 33.333...% → should round to 33.33
    df = pd.DataFrame({"employee_id": [1, 2, 3], "attrition": ["Yes", "No", "No"]})
    assert attrition_rate(df) == 33.33


# --- attrition_by_department ---

def test_attrition_by_department_columns(dept_df):
    result = attrition_by_department(dept_df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_counts_and_rates(dept_df):
    result = attrition_by_department(dept_df)
    sales = result.loc[result["department"] == "Sales"].iloc[0]
    hr = result.loc[result["department"] == "HR"].iloc[0]

    assert sales["employees"] == 3
    assert sales["leavers"] == 2
    assert sales["attrition_rate"] == round(2 / 3 * 100, 2)

    assert hr["employees"] == 3
    assert hr["leavers"] == 1
    assert hr["attrition_rate"] == round(1 / 3 * 100, 2)


def test_attrition_by_department_sorted_descending(dept_df):
    result = attrition_by_department(dept_df)
    rates = list(result["attrition_rate"])
    assert rates == sorted(rates, reverse=True)


# --- attrition_by_overtime ---

def test_attrition_by_overtime_columns():
    df = pd.DataFrame(
        {"employee_id": [1, 2], "overtime": ["Yes", "No"], "attrition": ["Yes", "No"]}
    )
    result = attrition_by_overtime(df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


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


# --- average_income_by_attrition ---

def test_average_income_by_attrition_columns():
    df = pd.DataFrame({"attrition": ["Yes", "No"], "monthly_income": [3000, 6000]})
    result = average_income_by_attrition(df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_values():
    df = pd.DataFrame(
        {
            "attrition": ["Yes", "Yes", "No", "No"],
            "monthly_income": [3000, 5000, 7000, 9000],
        }
    )
    result = average_income_by_attrition(df)
    assert result.loc[result["attrition"] == "Yes", "avg_monthly_income"].iloc[0] == 4000.0
    assert result.loc[result["attrition"] == "No", "avg_monthly_income"].iloc[0] == 8000.0


def test_average_income_by_attrition_rounds_to_two_decimals():
    # avg of 1000, 2000, 3001 = 2000.333... → should round to 2000.33
    df = pd.DataFrame(
        {"attrition": ["Yes", "Yes", "Yes"], "monthly_income": [1000, 2000, 3001]}
    )
    result = average_income_by_attrition(df)
    avg = result.loc[result["attrition"] == "Yes", "avg_monthly_income"].iloc[0]
    assert avg == round((1000 + 2000 + 3001) / 3, 2)


# --- satisfaction_summary ---

def test_satisfaction_summary_columns():
    df = pd.DataFrame(
        {"employee_id": [1, 2], "job_satisfaction": [1, 2], "attrition": ["Yes", "No"]}
    )
    result = satisfaction_summary(df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_attrition_rate_is_within_group():
    # 4 employees with satisfaction=1, 2 left → rate should be 50%
    # 2 employees with satisfaction=4, 0 left → rate should be 0%
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


def test_satisfaction_summary_sorted_ascending():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "job_satisfaction": [3, 1, 4, 2],
            "attrition": ["No", "Yes", "No", "No"],
        }
    )
    result = satisfaction_summary(df)
    scores = list(result["job_satisfaction"])
    assert scores == sorted(scores)
