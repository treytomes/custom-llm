import boto3
from datetime import date
from dataclasses import dataclass


@dataclass
class CostReport:
    current_month_cost: float
    forecast_month_end_cost: float


def get_cost_report() -> CostReport:
    ce = boto3.client("ce")

    today = date.today()
    start_of_month = today.replace(day=1)

    # first day of next month
    if today.month == 12:
        next_month = date(today.year + 1, 1, 1)
    else:
        next_month = date(today.year, today.month + 1, 1)

    usage = ce.get_cost_and_usage(
        TimePeriod={
            "Start": start_of_month.strftime("%Y-%m-%d"),
            "End": today.strftime("%Y-%m-%d"),
        },
        Granularity="MONTHLY",
        Metrics=["UnblendedCost"],
    )

    current_month_cost = float(
        usage["ResultsByTime"][0]["Total"]["UnblendedCost"]["Amount"]
    )

    forecast = ce.get_cost_forecast(
        TimePeriod={
            "Start": today.strftime("%Y-%m-%d"),
            "End": next_month.strftime("%Y-%m-%d"),
        },
        Metric="UNBLENDED_COST",
        Granularity="MONTHLY",
    )

    forecast_month_end = float(forecast["Total"]["Amount"])

    return CostReport(
        current_month_cost=current_month_cost,
        forecast_month_end_cost=forecast_month_end,
    )


if __name__ == "__main__":
    report = get_cost_report()
    print(f"Current month cost: ${report.current_month_cost:.2f}")
    print(f"Forecasted month end: ${report.forecast_month_end_cost:.2f}")