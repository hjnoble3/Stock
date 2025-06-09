import asyncio
import datetime
import math
from collections import defaultdict
from decimal import InvalidOperation

import nest_asyncio
import pandas as pd
from shared_tasty_utils import fetch_market_data_efficient
from tastytrade import instruments, metrics
from tastytrade.market_data import get_market_data_by_type

nest_asyncio.apply()

DAYS_IN_YEAR = 365.0
FIVE_DECIMAL_PLACES = 5
DEFAULT_EQUITY_MULTIPLIER = 100.0


def calculateBetaWeightedDelta(delta, beta_val, current_price, spy_price):
    if beta_val is not None and spy_price and spy_price != 0 and current_price:
        return delta * beta_val * (current_price / spy_price)
    return None


def calculateExpectedMove(current_price, expiration_iv, days_to_expiration, sqrt_days_cache):
    if expiration_iv is not None and current_price is not None and days_to_expiration > 0:
        sqrt_term = sqrt_days_cache.get(days_to_expiration)
        if sqrt_term is None:
            sqrt_term = math.sqrt(days_to_expiration / DAYS_IN_YEAR)
            sqrt_days_cache[days_to_expiration] = sqrt_term
        return current_price * expiration_iv * sqrt_term
    return None


async def fetchUnderlyingAndSpyPrices(session, symbol):
    instrument_kwarg = {"futures": [symbol]} if symbol.startswith("/") else {"equities": [symbol]}
    market_data_underlying = get_market_data_by_type(session, **instrument_kwarg)
    if not market_data_underlying or market_data_underlying[0].mid is None:
        print(f"Error: Could not fetch valid market data (mid price) for {symbol}.")
        return None, None
    current_price = float(market_data_underlying[0].mid)

    if symbol == "SPY":
        return current_price, current_price

    spy_data = get_market_data_by_type(session, indices=["SPY"])
    if not spy_data or spy_data[0].mid is None:
        print("Warning: Could not fetch valid market data for SPY. Beta-weighted delta may be None.")
        return current_price, None

    return current_price, float(spy_data[0].mid)


async def fetchMarketMetrics(session, symbol):
    try:
        market_metrics_data = metrics.get_market_metrics(session, [symbol])
    except Exception as e:
        print(f"Error fetching market metrics for {symbol}: {e}")
        return None, {}

    if not market_metrics_data:
        print(f"Error: Could not fetch market metrics for {symbol}.")
        return None, {}

    metric_info = market_metrics_data[0]
    beta_val = float(metric_info.beta) if metric_info.beta is not None else None

    iv_lookup = {}
    for iv_data in metric_info.option_expiration_implied_volatilities or []:
        if iv_data and iv_data.expiration_date and iv_data.implied_volatility is not None:
            try:
                iv_lookup[iv_data.expiration_date] = float(iv_data.implied_volatility)
            except InvalidOperation:
                print(f"Warning: Could not convert IV '{iv_data.implied_volatility}' to float for {iv_data.expiration_date}.")

    return beta_val, iv_lookup


async def fetchAndFilterOptionChain(session, symbol, min_days, max_days):
    print(f"Fetching option chain for {symbol}...")
    chain = await (
        instruments.a_get_future_option_chain(session, symbol) if symbol.startswith("/") else instruments.a_get_option_chain(session, symbol)
    )
    if not chain:
        print(f"Error: Could not fetch option chain for {symbol}.")
        return defaultdict(list)

    today = datetime.date.today()
    return defaultdict(
        list,
        {expiry: opts for expiry, opts in chain.items() if isinstance(expiry, datetime.date) and min_days <= (expiry - today).days <= max_days},
    )


def filterOptionsByCriteria(all_symbol_data, min_open_interest, min_abs_delta_dec, slippage_threshold_dec, current_price):
    matching_symbols_data = []
    for symbol_str, data in all_symbol_data.items():
        summary = data.get("summary")
        greeks = data.get("greeks")
        quote = data.get("quote")

        if not (summary and greeks and quote):
            continue

        open_interest = getattr(summary, "open_interest", 0)
        delta = float(getattr(greeks, "delta", 0))
        bid = getattr(quote, "bid_price", None)
        ask = getattr(quote, "ask_price", None)

        if open_interest >= min_open_interest and abs(delta) >= min_abs_delta_dec and bid is not None and ask is not None:
            slippage = (float(ask) - float(bid)) / current_price if current_price != 0 else None
            if slippage is not None and slippage <= slippage_threshold_dec:
                matching_symbols_data.append((symbol_str, data))

    print(f"Found {len(matching_symbols_data)} matching symbols after filtering.")
    return matching_symbols_data


async def getFilteredOptionData(
    session,
    symbol,
    min_days=4,
    max_days=85,
    min_open_interest=100,
    min_abs_delta=0.10,
    slippage_threshold=0.05,
):
    print(f"Starting data retrieval for {symbol}...")
    sqrt_days_cache = {}

    notional_multiplier = (await instruments.get_future(session, symbol)).notional_multiplier if symbol.startswith("/") else DEFAULT_EQUITY_MULTIPLIER

    tasks = await asyncio.gather(
        fetchUnderlyingAndSpyPrices(session, symbol),
        fetchMarketMetrics(session, symbol),
        fetchAndFilterOptionChain(session, symbol, min_days, max_days),
    )
    (current_price, spy_price), (beta_val, iv_lookup), filtered_chain = tasks

    if current_price is None:
        return pd.DataFrame(), None, None
    if beta_val is None and iv_lookup == {}:
        return pd.DataFrame(), current_price, spy_price
    if not filtered_chain:
        return pd.DataFrame(), current_price, spy_price

    all_options = [opt for opts in filtered_chain.values() for opt in opts]
    streamer_symbols = [opt.streamer_symbol for opt in all_options]
    option_lookup = {opt.streamer_symbol: opt for opt in all_options}

    print(f"Collected {len(streamer_symbols)} streamer symbols for market data.")
    print("Fetching market data for all relevant options...")
    all_symbol_data = await fetch_market_data_efficient(session, streamer_symbols)

    matching_symbols_data = filterOptionsByCriteria(all_symbol_data, min_open_interest, min_abs_delta, slippage_threshold, current_price)

    today = datetime.date.today()
    final_enriched_data = []

    for streamer_symbol, market_data in matching_symbols_data:
        option_obj = option_lookup.get(streamer_symbol)
        if not option_obj:
            continue

        summary = market_data.get("summary")
        greeks = market_data.get("greeks")
        quote = market_data.get("quote")

        bid = float(quote.bid_price) if quote.bid_price is not None else None
        ask = float(quote.ask_price) if quote.ask_price is not None else None
        mid_price = (bid + ask) / 2 if bid and ask else None

        expiration_iv = iv_lookup.get(option_obj.expiration_date)
        if expiration_iv is None:
            continue

        days_to_expiration = (option_obj.expiration_date - today).days
        if days_to_expiration <= 0:
            continue

        delta = float(greeks.delta) if greeks.delta else 0.0
        beta_delta = calculateBetaWeightedDelta(delta, beta_val, current_price, spy_price)
        std_dev = calculateExpectedMove(current_price, expiration_iv, days_to_expiration, sqrt_days_cache)

        enriched = {
            "streamer_symbol": streamer_symbol,
            "underlying_symbol": option_obj.underlying_symbol,
            "expiration_date": option_obj.expiration_date,
            "option_type": option_obj.option_type.value,
            "strike_price": option_obj.strike_price,
            "delta": round(delta, FIVE_DECIMAL_PLACES),
            "theta": round(float(greeks.theta), FIVE_DECIMAL_PLACES) if greeks.theta is not None else None,
            "mid_price": round(mid_price, FIVE_DECIMAL_PLACES) if mid_price is not None else None,
            "beta": round(beta_val, FIVE_DECIMAL_PLACES) if beta_val is not None else None,
            "notional_multiplier": notional_multiplier,
            "beta_weighted_delta": round(beta_delta, FIVE_DECIMAL_PLACES) if beta_delta is not None else None,
            "standard_deviation": round(std_dev, FIVE_DECIMAL_PLACES) if std_dev is not None else None,
            "days_to_expiration": days_to_expiration,
        }

        final_enriched_data.append(enriched)

    df = pd.DataFrame(final_enriched_data)
    print(f"Final DataFrame created with {len(df)} rows.")
    return df, current_price
