import asyncio

import nest_asyncio
import numpy as np
import pandas as pd
from chain import getFilteredOptionData
from OptionStrategies.broken_wing_butterflys import create_broken_wing_butterflys
from OptionStrategies.calendar_spreads import create_calendar_spreads
from OptionStrategies.diagonal_spreads import create_diagonal_spreads
from OptionStrategies.iron_condors import create_iron_condors

# Import strategy creation functions from the new subfolder
from OptionStrategies.spreads import create_spreads
from OptionStrategies.strangles import create_strangles
from session_login import session_login
from shared_tasty_utils import (
    calculate_cvar,
    calculate_probability_of_profit,
)

# Apply nested asyncio for compatibility
nest_asyncio.apply()

# Configuration constants
SYMBOL = "TSLA"
MIN_DAYS = 4
MAX_DAYS = 85
MIN_OPEN_INTEREST = 100
MIN_ABS_DELTA = 0.10
SLIPPAGE_THRESHOLD = 0.05
CONFIDENCE_LEVEL = 0.95  # For CVaR calculation (e.g., 95% means we care about the worst 5% of outcomes)


async def fetch_option_data(session):
    """Fetches option data for a given symbol and applies filtering criteria."""
    df, current_price = await getFilteredOptionData(
        session=session,
        symbol=SYMBOL,
        min_days=MIN_DAYS,
        max_days=MAX_DAYS,
        min_open_interest=MIN_OPEN_INTEREST,
        min_abs_delta=MIN_ABS_DELTA,
        slippage_threshold=SLIPPAGE_THRESHOLD,
    )
    print(f"Retrieved {len(df)} options for {SYMBOL} matching the criteria.")
    return df, current_price


def create_positions(df, current_price):
    """Creates individual long and short option positions from the fetched data,
    calculating their key metrics including CVaR.
    """
    numerical_columns = ["strike_price", "mid_price", "notional_multiplier", "delta", "theta", "beta_weighted_delta", "standard_deviation"]
    for col in numerical_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["expiration_date"] = pd.to_datetime(df["expiration_date"])

    # Long positions (e.g., Long Call, Long Put)
    long_df = df[df["mid_price"] <= 5.0].copy()  # Example filter for cheap options to consider long
    long_df["strategy"] = long_df["option_type"].map({"C": "Long Call", "P": "Long Put"})
    long_df["position_type"] = long_df["strategy"]
    long_df["lower_breakeven"] = np.where(long_df["option_type"] == "P", long_df["strike_price"] - long_df["mid_price"], np.nan)
    long_df["upper_breakeven"] = np.where(long_df["option_type"] == "C", long_df["strike_price"] + long_df["mid_price"], np.nan)
    long_df["max_profit"] = np.inf  # Long options have theoretically unlimited profit
    long_df["probability_of_profit"] = long_df.apply(
        lambda row: calculate_probability_of_profit(
            current_price,
            row["upper_breakeven"] if row["option_type"] == "C" else row["lower_breakeven"],
            row["standard_deviation"],
            row["strategy"],
        ),
        axis=1,
    )
    long_df["cvar"] = long_df.apply(
        lambda row: calculate_cvar(
            row["strategy"],
            row["mid_price"],
            row["notional_multiplier"],
            pop=row["probability_of_profit"],
            current_price=current_price,
            standard_deviation=row["standard_deviation"],
        ),
        axis=1,
    )

    # Short positions (e.g., Short Call, Short Put)
    short_df = df.copy()
    short_df["strategy"] = short_df["option_type"].map({"C": "Short Call", "P": "Short Put"})
    short_df["position_type"] = short_df["strategy"]
    short_df["lower_breakeven"] = np.where(short_df["option_type"] == "P", short_df["strike_price"] - short_df["mid_price"], np.nan)
    short_df["upper_breakeven"] = np.where(short_df["option_type"] == "C", short_df["strike_price"] + short_df["mid_price"], np.nan)
    short_df["max_profit"] = short_df["mid_price"] * short_df["notional_multiplier"]  # Max profit is premium received
    # Invert delta, theta, beta_weighted_delta for short positions
    short_df[["delta", "theta", "beta_weighted_delta"]] = -short_df[["delta", "theta", "beta_weighted_delta"]]
    short_df["probability_of_profit"] = short_df.apply(
        lambda row: calculate_probability_of_profit(
            current_price,
            row["upper_breakeven"] if row["option_type"] == "C" else row["lower_breakeven"],
            row["standard_deviation"],
            row["strategy"],
        ),
        axis=1,
    )
    short_df["cvar"] = short_df.apply(
        lambda row: calculate_cvar(
            row["strategy"],
            row["mid_price"],
            row["notional_multiplier"],
            strike_leg1=row["strike_price"],  # Pass strike price for CVaR calculation
            pop=row["probability_of_profit"],
            current_price=current_price,
            standard_deviation=row["standard_deviation"],
        ),
        axis=1,
    )
    # Filter for short positions with a minimum probability of profit
    short_df = short_df[short_df["probability_of_profit"] >= 0.68]

    # Combine long and short positions
    positions_df = pd.concat([long_df, short_df], ignore_index=True)
    return positions_df[
        [
            "expiration_date",
            "option_type",
            "strategy",
            "strike_price",
            "mid_price",
            "max_profit",
            "delta",
            "theta",
            "beta_weighted_delta",
            "position_type",
            "lower_breakeven",
            "upper_breakeven",
            "probability_of_profit",
            "cvar",
            "notional_multiplier",
            "standard_deviation",
        ]
    ]


async def process_option_strategies():
    """Main function to fetch data, create positions, spreads, strangles, iron condors,
    broken wing butterflys, calendar spreads, and diagonal spreads.
    """
    session = session_login()
    df, current_price = await fetch_option_data(session)
    if df.empty:
        print("No option data retrieved. Exiting.")
        # Ensure all expected return values are provided as empty lists/DataFrame
        return pd.DataFrame(), [], [], [], [], [], []

    positions_df = create_positions(df, current_price)
    grouped_spreads = positions_df.groupby(["expiration_date", "option_type"])
    grouped_expirations = positions_df.groupby("expiration_date")

    spreads_list = create_spreads(grouped_spreads, current_price)
    strangles_list = create_strangles(grouped_expirations, current_price)
    iron_condors_list = create_iron_condors(grouped_expirations, current_price)
    broken_wing_butterflys_list = create_broken_wing_butterflys(grouped_expirations, current_price)
    calendar_spreads_list = create_calendar_spreads(df, current_price)  # Calendars use raw df to find different expirations
    diagonal_spreads_list = create_diagonal_spreads(df, current_price)  # Diagonals use raw df

    return positions_df, spreads_list, strangles_list, iron_condors_list, broken_wing_butterflys_list, calendar_spreads_list, diagonal_spreads_list


if __name__ == "__main__":
    # Update variable assignment to include new lists
    positions_df, spreads_list, strangles_list, iron_condors_list, broken_wing_butterflys_list, calendar_spreads_list, diagonal_spreads_list = (
        asyncio.run(process_option_strategies())
    )
    print("\nOption data processing completed.")

    print("\n--- Positions ---")
    if not positions_df.empty:
        print(positions_df.head())
        print(f"\nTotal positions created: {len(positions_df)}")
    else:
        print("No positions created.")

    print("\n--- Spreads ---")
    if spreads_list:
        spreads_df = pd.DataFrame(spreads_list)
        print(spreads_df.head())
        print(f"\nTotal spreads created: {len(spreads_df)}")
    else:
        print("No spreads created.")

    print("\n--- Strangles ---")
    if strangles_list:
        strangles_df = pd.DataFrame(strangles_list)
        print(strangles_df.head())
        print(f"\nTotal strangles created: {len(strangles_df)}")
    else:
        print("No strangles created.")

    print("\n--- Iron Condors ---")
    if iron_condors_list:
        iron_condors_df = pd.DataFrame(iron_condors_list)
        print(iron_condors_df.head())
        print(f"\nTotal iron condors created: {len(iron_condors_df)}")
    else:
        print("No iron condors created.")

    print("\n--- Broken Wing Butterflys ---")
    if broken_wing_butterflys_list:
        broken_wing_butterflys_df = pd.DataFrame(broken_wing_butterflys_list)
        print(broken_wing_butterflys_df.head())
        print(f"\nTotal broken wing butterflys created: {len(broken_wing_butterflys_df)}")
    else:
        print("No broken wing butterflys created.")

    print("\n--- Calendar Spreads ---")
    if calendar_spreads_list:
        calendar_spreads_df = pd.DataFrame(calendar_spreads_list)
        print(calendar_spreads_df.head())
        print(f"\nTotal calendar spreads created: {len(calendar_spreads_list)}")
    else:
        print("No calendar spreads created.")

    print("\n--- Diagonal Spreads ---")
    if diagonal_spreads_list:
        diagonal_spreads_df = pd.DataFrame(diagonal_spreads_list)
        print(diagonal_spreads_df.head())
        print(f"\nTotal diagonal spreads created: {len(diagonal_spreads_list)}")
    else:
        print("No diagonal spreads created.")
