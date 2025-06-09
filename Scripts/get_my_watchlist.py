from datetime import datetime, timedelta

import pandas as pd
from dateutil.relativedelta import relativedelta
from session_login import session_login
from shared_tasty_utils import (
    build_instrument_args,
    flatten_market_metric_info,
)
from tastytrade import metrics
from tastytrade.market_data import get_market_data_by_type
from tastytrade.watchlists import PrivateWatchlist

# Constants
WATCHLIST_NAME = "MyWatchlist"

# Month codes dictionary
month_codes = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M", 7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}


def get_next_four_months():
    """Calculates the next four valid futures expiration months based on the current date,
    ensuring they are at least 14 days in the future.
    """
    today = datetime.now()  # Get current datetime
    min_expiration_date = today + timedelta(days=14)
    next_months = []
    months_ahead = 0

    while len(next_months) < 4:
        future_date = today + relativedelta(months=months_ahead)
        month = future_date.month
        year = future_date.year % 10  # Get the last digit of the year
        # Calculate the last day of the future_date's month
        expiration_date = future_date.replace(day=1) + relativedelta(months=1) - timedelta(days=1)

        if month in month_codes and expiration_date >= min_expiration_date:
            next_months.append(f"{month_codes[month]}{year}")

        months_ahead += 1

    return next_months


def append_futures_months(df: pd.DataFrame) -> pd.DataFrame:
    """Appends futures expiration months to underlying symbols in the DataFrame
    where applicable (symbols starting with '/' but not '/BTC' or '/ETH').
    """
    futures_months = get_next_four_months()

    def process_symbol(symbol: str) -> list[str]:
        """Helper function to generate futures symbols or return original symbol."""
        if symbol.startswith("/") and symbol not in ["/BTC", "/ETH"]:
            return [f"{symbol}{month}" for month in futures_months]
        return [symbol]

    df["underlying_symbol"] = df["underlying_symbol"].apply(process_symbol)
    # Explode the list of symbols into separate rows
    return df.explode("underlying_symbol").reset_index(drop=True)


def process_tastytrade_data() -> pd.DataFrame:
    """Main function to log in, fetch watchlist data, retrieve market data and metrics,
    process them, and return a merged DataFrame.
    """
    # 1. Login and fetch watchlist
    session = session_login()
    watchlist_data = PrivateWatchlist.get(session, WATCHLIST_NAME)
    df_watchlist = pd.DataFrame(watchlist_data.watchlist_entries).rename(
        columns={"symbol": "underlying_symbol", "instrument-type": "instrument_type"},
    )
    df_watchlist = append_futures_months(df_watchlist)

    # 2. Market data processing
    market_data_args = build_instrument_args(df_watchlist)
    market_data = []
    chunk_size = 100

    # Iterate through instrument types and fetch market data in chunks
    for instrument_type in ["indices", "equities", "futures"]:
        instruments_to_fetch = market_data_args.get(instrument_type, [])
        for i in range(0, len(instruments_to_fetch), chunk_size):
            chunk = instruments_to_fetch[i : i + chunk_size]
            # Call the API with the specific instrument type chunk
            chunk_data = get_market_data_by_type(
                session,
                indices=chunk if instrument_type == "indices" else [],
                equities=chunk if instrument_type == "equities" else [],
                futures=chunk if instrument_type == "futures" else [],
            )
            market_data.extend(chunk_data)

    # 3. Create market data DataFrame
    market_data_dicts = [
        {
            "symbol": data.symbol,
            "instrument_type": str(data.instrument_type).split(".")[-1],
            "mark": float(data.mark),
            # Round beta to 2 decimal places
            "beta": round(float(data.beta), 2) if data.beta is not None else None,
            "daily_change": round(float(data.mark) - float(data.prev_close), 2),
            "year_low_price": float(data.year_low_price) if data.year_low_price is not None else None,
            "year_high_price": float(data.year_high_price) if data.year_high_price is not None else None,
        }
        for data in market_data
    ]
    df_market_data = pd.DataFrame(market_data_dicts)

    # 4. Calculate metrics
    # Calculate percentage change based on daily change
    df_market_data["pct_change"] = (df_market_data["daily_change"] / (df_market_data["mark"] - df_market_data["daily_change"]) * 100).round(2)

    # Calculate relative position within 52-week high/low range
    # Renamed from 'relative_position' to 'relative_to_52_week_range_pct'
    df_market_data["relative_to_52_week_range_pct"] = (
        ((df_market_data["mark"] - df_market_data["year_low_price"]) / (df_market_data["year_high_price"] - df_market_data["year_low_price"]) * 100)
        .round(2)
        .where(
            # Condition to avoid division by zero and handle missing data
            (df_market_data["year_high_price"] != df_market_data["year_low_price"])
            & df_market_data[["year_high_price", "year_low_price", "mark"]].notnull().all(axis=1),
            None,  # Set to None if conditions are not met
        )
    )

    # 5. Fetch and process additional metrics
    metrics_list = metrics.get_market_metrics(session, df_market_data["symbol"].unique().tolist())
    metric_df = pd.DataFrame([flatten_market_metric_info(m) for m in metrics_list])

    # Select only required columns and adjust implied_volatility_percentile
    cols = [
        "symbol",
        "implied_volatility_percentile",
        "implied_volatility_index_5_day_change",
        "implied_volatility_index_rank",
        "corr_spy_3month",
        "dividend_ex_date",
        "dividend_pay_date",
    ]
    # Add earnings columns if they exist in the metric_df
    cols.extend([col for col in ["earnings_expected_report_date", "earnings_time_of_day"] if col in metric_df.columns])
    metric_df = metric_df[cols]

    # Ensure implied volatility columns are numeric before calculations, then apply * 100 and round to two decimals
    for col_name in ["implied_volatility_percentile", "implied_volatility_index_5_day_change", "implied_volatility_index_rank"]:
        if col_name in metric_df.columns:
            # Convert column to numeric, coercing any errors to NaN
            metric_df[col_name] = pd.to_numeric(metric_df[col_name], errors="coerce")
            # Perform multiplication and rounding
            metric_df[col_name] = (metric_df[col_name] * 100).round(2)

    # 6. Process date columns to "days until"
    # Get the current date and time with the specified timezone
    current_date = pd.Timestamp.now(tz="America/New_York")
    date_cols = [col for col in ["dividend_ex_date", "dividend_pay_date", "earnings_expected_report_date"] if col in metric_df.columns]

    for col in date_cols:
        # Convert date columns to datetime objects, coercing errors
        metric_df[col] = pd.to_datetime(metric_df[col], errors="coerce", utc=True)
        # Calculate days until, setting to None if date is in the past or NaT
        metric_df[col] = (metric_df[col] - current_date.tz_convert("UTC")).dt.days.where(
            metric_df[col].notna() & (metric_df[col] >= current_date.tz_convert("UTC")),
            None,
        )

    # 7. Create merge key for df_market_data
    # For futures, the merge key is the first 3 characters (e.g., '/ES' for '/ESZ24')
    df_market_data["merge_key"] = df_market_data["symbol"].apply(lambda x: x[:3] if x.startswith("/") else x)

    # 8. Rename metric_df's symbol to underlying_symbol for merge
    metric_df = metric_df.rename(columns={"symbol": "underlying_symbol"})

    # 9. Merge DataFrames
    # Merge market data with metrics based on the derived underlying symbol
    merged_df = df_market_data.merge(metric_df, how="left", left_on="merge_key", right_on="underlying_symbol")

    # 10. Drop unwanted columns
    merged_df = merged_df.drop(columns=["merge_key", "underlying_symbol"])

    return merged_df


# Main execution block
if __name__ == "__main__":
    # Call the main processing function
    final_merged_df = process_tastytrade_data()
    # Display the first few rows of the final DataFrame
    print(final_merged_df.head())
