import asyncio
from decimal import Decimal

import pandas as pd
from session_login import session_login
from shared_tasty_utils import (
    build_instrument_args,
    fetch_greek_data,
    fetch_option_data,
    flatten_market_metric_info,
)
from tastytrade import Account, metrics
from tastytrade.market_data import get_market_data_by_type


async def get_positions_and_account_info(session):
    """Retrieves position-level and account-level information for all accounts in a Tastytrade session.
    Each position (leg) is preserved without rolling up to the underlying symbol.

    Args:
        session: Tastytrade session object from session_login.

    Returns:
        List of dictionaries, each containing:
            - account_nickname: Account nickname.
            - result_df: DataFrame with position-level data (one row per leg).
            - Account metrics (bpr_usage, bpr_left_to_use, net_liquidating_left, max_delta, max_theta, min_theta).

    """
    # Cache VIX and SPY prices (single API call)
    market_data = get_market_data_by_type(session, indices=["VIX", "SPY"])
    vix = Decimal(str(market_data[0].mark if market_data[0].symbol == "VIX" else market_data[1].mark))
    spy_price = Decimal(str(market_data[1].mark if market_data[1].symbol == "SPY" else market_data[0].mark))

    # Define VIX ranges for BPR calculation
    vix_ranges = [
        (Decimal("0"), Decimal("15"), Decimal("0.25")),
        (Decimal("15"), Decimal("20"), Decimal("0.30")),
        (Decimal("20"), Decimal("30"), Decimal("0.35")),
        (Decimal("30"), Decimal("40"), Decimal("0.40")),
        (Decimal("40"), Decimal(str(float("inf"))), Decimal("0.50")),
    ]
    max_bpr = next(multiplier for lower, upper, multiplier in vix_ranges if lower < vix <= upper)

    # Fetch all accounts once
    accounts = Account.get(session)
    all_accounts_data = []

    # Get today's date in UTC for timezone consistency
    today = pd.Timestamp.today(tz="UTC").normalize()

    for account in accounts:
        balance = account.get_balances(session)
        if balance.net_liquidating_value <= 0:
            continue

        # Account-level calculations
        account_nickname = account.nickname
        net_liquidating_value = Decimal(str(balance.net_liquidating_value))
        margin_requirements = account.get_margin_requirements(session)
        net_liquidating_used = Decimal(str(abs(margin_requirements.margin_requirement)))
        net_liquidating_percent = net_liquidating_used / net_liquidating_value
        account_metrics = {
            "bpr_usage": net_liquidating_percent > max_bpr,
            "bpr_left_to_use": (max_bpr - net_liquidating_percent) * net_liquidating_value,
            "net_liquidating_left": Decimal(str(margin_requirements.option_buying_power)),
            "max_delta": net_liquidating_value / spy_price,
            "max_theta": net_liquidating_value * Decimal("0.003"),
            "min_theta": net_liquidating_value * Decimal("0.001"),
        }

        # Fetch positions and margin requirements
        positions = account.get_positions(session)
        position_margin_requirements = [
            {"underlying_symbol": entry.underlying_symbol, "margin_requirement": Decimal(str(abs(entry.margin_requirement)))}
            for entry in margin_requirements.groups
        ]

        # Create positions DataFrame with robust datetime handling
        positions_df = pd.DataFrame(
            [
                {
                    "underlying_symbol": p.underlying_symbol,
                    "symbol": p.symbol,
                    "instrument_type": p.instrument_type.value if hasattr(p.instrument_type, "value") else str(p.instrument_type),
                    "quantity": Decimal(str(p.quantity)) * (-1 if p.quantity_direction == "Short" else 1),
                    "average_open_price": Decimal(str(p.average_open_price)),
                    "multiplier": Decimal(str(p.multiplier)),
                    "expires_at": pd.to_datetime(p.expires_at, errors="coerce", utc=True),
                }
                for p in positions
            ],
        )

        if positions_df.empty:
            continue

        # Convert expires_at to days until expiration
        positions_df["days_to_expiration"] = (
            (positions_df["expires_at"] - today).dt.days
            if positions_df["expires_at"].notna().any()
            else pd.Series(pd.NA, index=positions_df.index, dtype="object")
        )
        positions_df["days_to_expiration"] = positions_df["days_to_expiration"].where(positions_df["days_to_expiration"].notna(), pd.NA)

        # Fetch market data
        market_data_args = build_instrument_args(positions_df)
        market_data = get_market_data_by_type(
            session,
            indices=market_data_args["indices"],
            cryptocurrencies=market_data_args["cryptocurrencies"],
            equities=market_data_args["equities"],
            futures=market_data_args["futures"],
            future_options=market_data_args["future_options"],
            options=market_data_args["options"],
        )
        market_df = pd.DataFrame(
            [
                {
                    "symbol": d.symbol,
                    "mid": Decimal(str(d.mid)),
                    "prev_close": Decimal(str(d.prev_close)),
                    "daily_change": Decimal(str(d.prev_close - d.mid)),
                }
                for d in market_data
            ],
        )

        # Merge positions and market data
        result_df = positions_df.merge(market_df, on="symbol", how="left")

        # Fetch option data and greeks for non-equity positions
        non_equity_symbols = result_df[result_df["instrument_type"] != "Equity"]["underlying_symbol"].unique()
        if non_equity_symbols.size > 0:
            symbol_to_streamer = await fetch_option_data(session, non_equity_symbols.tolist())
            result_df["streamer_symbol"] = result_df["symbol"].map(symbol_to_streamer)
            streamer_symbols = result_df["streamer_symbol"].dropna().unique().tolist()
            if streamer_symbols:
                greek_df = await fetch_greek_data(session, streamer_symbols)
                result_df = result_df.merge(greek_df, on="streamer_symbol", how="left")
        else:
            result_df["streamer_symbol"] = pd.NA

        # Ensure greek columns exist and apply multiplier and quantity to delta and theta
        for col in ["delta", "gamma", "theta", "rho", "vega", "price", "volatility"]:
            result_df[col] = result_df.get(col, pd.NA)
        result_df["delta"] = (result_df["delta"] * result_df["multiplier"] * result_df["quantity"]).fillna(Decimal("1.0"))
        result_df["theta"] = result_df["theta"] * result_df["multiplier"] * result_df["quantity"]
        result_df[["delta", "theta"]] = result_df[["delta", "theta"]].apply(
            lambda x: x.apply(lambda y: y.quantize(Decimal("0.01")) if pd.notna(y) and isinstance(y, Decimal) else y),
        )

        # Fetch market metrics
        metrics_list = metrics.get_market_metrics(session, result_df["underlying_symbol"].unique().tolist())
        metric_df = pd.DataFrame([flatten_market_metric_info(m) for m in metrics_list])
        cols = [
            "symbol",
            "implied_volatility_index_rank",
            "implied_volatility_percentile",
            "beta",
            "corr_spy_3month",
            "dividend_ex_date",
            "dividend_pay_date",
        ]
        cols.extend([col for col in ["earnings_expected_report_date", "earnings_time_of_day"] if col in metric_df.columns])
        metric_df = metric_df[cols].rename(columns={"symbol": "underlying_symbol"})

        # Process date columns
        date_cols = [col for col in ["dividend_ex_date", "dividend_pay_date", "earnings_expected_report_date"] if col in metric_df.columns]
        for col in date_cols:
            metric_df[col] = pd.to_datetime(metric_df[col], errors="coerce", utc=True)
            metric_df[col] = metric_df[col].where(metric_df[col].isna() | (metric_df[col] >= today))

        # Merge metrics and calculate position-level metrics
        result_df = result_df.merge(metric_df, on="underlying_symbol", how="left")
        result_df = result_df.drop(
            columns=["mid", "prev_close", "daily_change", "multiplier", "streamer_symbol", "volatility", "gamma", "rho", "vega"],
            errors="ignore",
        )
        result_df["open_price"] = result_df["average_open_price"] * result_df["quantity"]
        result_df["current_price"] = result_df["price"] * result_df["quantity"]
        result_df["profit"] = result_df["open_price"] - result_df["current_price"]
        result_df["percent_profit"] = (result_df["profit"] / result_df["open_price"] * Decimal("100")).where(
            result_df["open_price"] != 0,
            Decimal("0"),
        )

        # Fetch and merge underlying market data
        market_underlying_args = build_instrument_args(result_df)
        market_data = get_market_data_by_type(
            session,
            indices=market_underlying_args["indices"],
            cryptocurrencies=market_underlying_args["cryptocurrencies"],
            equities=market_underlying_args["equities"],
            futures=market_underlying_args["futures"],
            future_options=market_underlying_args["future_options"],
            options=market_underlying_args["options"],
        )
        market_df = pd.DataFrame(
            [
                {
                    "underlying_symbol": d.symbol,
                    "mid": Decimal(str(d.mid)),
                    "daily_change": Decimal(str(d.prev_close - d.mid)),
                }
                for d in market_data
            ],
        )
        market_df["prev_close"] = market_df["mid"] - market_df["daily_change"]
        market_df["pct_change"] = (market_df["daily_change"] / market_df["prev_close"] * Decimal("-100")).apply(
            lambda x: x.quantize(Decimal("0.01")) if pd.notna(x) and isinstance(x, Decimal) else x,
        )

        # Final merge and calculations
        result_df = result_df.merge(market_df, on="underlying_symbol", how="left")
        result_df = result_df.merge(pd.DataFrame(position_margin_requirements), on="underlying_symbol", how="left")
        result_df["beta_weighted_delta"] = (result_df["delta"] * result_df["beta"] * (result_df["mid"] / spy_price)).apply(
            lambda x: x.quantize(Decimal("0.01")) if pd.notna(x) and isinstance(x, Decimal) else x,
        )

        # Round all Decimal/float columns to 2 decimal places
        decimal_cols = result_df.select_dtypes(include=["float64", "object"]).columns
        decimal_cols = [col for col in decimal_cols if result_df[col].apply(lambda x: isinstance(x, Decimal)).any()]
        result_df[decimal_cols] = result_df[decimal_cols].apply(
            lambda x: x.apply(lambda y: y.quantize(Decimal("0.01")) if pd.notna(y) and isinstance(y, Decimal) else y),
        )

        # Store account data
        all_accounts_data.append(
            {
                "account_nickname": account_nickname,
                "result_df": result_df,
                **account_metrics,
            },
        )

    return all_accounts_data


# Example usage
if __name__ == "__main__":
    session = session_login()
    all_accounts_data = asyncio.run(get_positions_and_account_info(session))
    for account_data in all_accounts_data:
        print(f"Account: {account_data['account_nickname']}")
        print(account_data["result_df"])
