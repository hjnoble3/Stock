import asyncio
import datetime
import logging
import os
import statistics
import sys
from decimal import Decimal

import pandas as pd
from tastytrade import DXLinkStreamer, instruments, metrics
from tastytrade.dxfeed import Quote
from tastytrade.market_data import get_market_data_by_type
from tastytrade.watchlists import PrivateWatchlist, PublicWatchlist
from tqdm import tqdm

# Add parent directory to sys.path for imports
_CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.abspath(os.path.join(_CURRENT_FILE_DIR, ".."))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from session_login import session_login
from shared_tasty_utils import (
    MAX_WORKERS,
    REQUEST_TIMEOUT,
    fetch_market_data_efficient,
    get_option_chain_async,
)

# --- Configuration Constants ---
MIN_LIQUIDITY_NON_ETF = 2
MIN_LIQUIDITY_ETF = 3
MIN_EXPIRATION_DAYS_UNIVERSE = 5
MAX_EXPIRATION_DAYS_UNIVERSE = 83
MIN_EXPIRATIONS_COUNT_UNIVERSE = 5

DELTA_THRESHOLD = Decimal("0.16")
MIN_OPEN_INTEREST = 100
MIN_STOCK_PRICE = 5
MIN_VALID_OPTIONS = 6
MIN_EXPIRATIONS_COUNT_TRADABLE = 3
MAX_AVG_SPREAD_PERCENTAGE = 0.5
VIX_SYMBOL = "VIX"
WATCHLIST_NAME = "MyWatchlist"

FORCED_TICKERS = ["SPY", "QQQ"]
FORCED_REMOVE_TICKERS = ["/BTC", "/ETH"]  # These will now be treated as prefixes

DATA_DIR = os.path.join(_SCRIPTS_DIR, "Data")
CSV_OUTPUT_DIR = os.path.join(DATA_DIR, "Filter_CSVs")
TICKER_REMOVAL_FILE = os.path.join(DATA_DIR, "ticker_removal_watch.txt")
TICKER_ADD_FILE = os.path.join(DATA_DIR, "ticker_add_watch.txt")

ESSENTIAL_COLUMNS = [
    "entry",
    "stock_price",
    "avg_spread_percentage",
    "median_spread_percentage",
    "valid_options",
    "avg_options_per_expiration",
    "expirations_count",
]

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s",
)


# --- Helper Functions ---
def initialize_directories_and_files():
    """Ensures data and CSV directories exist and creates empty ticker files if needed."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
    for file_path in [TICKER_REMOVAL_FILE, TICKER_ADD_FILE]:
        if not os.path.exists(file_path):
            open(file_path, "a").close()
            logging.info(f"Created empty file: {file_path}")


def read_ticker_file(path):
    """Reads tickers from a file, returning a list of stripped strings."""
    try:
        with open(path) as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        logging.warning(f"File {path} not found.")
        return []


def write_ticker_file(path, tickers):
    """Writes sorted tickers to a file, one per line."""
    with open(path, "w") as file:
        file.writelines(f"{ticker}\n" for ticker in sorted(tickers))
    logging.info(f"Updated file: {path} with {len(tickers)} tickers")


def save_dataframe_to_csv(df, filename_prefix, stage_name):
    """Saves a DataFrame to a CSV file with specified columns."""
    if filename_prefix == "ticker_universe" and stage_name == "expirations_filtered":
        columns_to_save = ["symbol", "num_valid_expirations_universe"]
    else:
        columns_to_save = [col for col in ESSENTIAL_COLUMNS if col in df.columns]
    if not columns_to_save:
        logging.warning(f"No valid columns for CSV: {filename_prefix}_{stage_name}")
        return
    file_path = os.path.join(CSV_OUTPUT_DIR, f"{filename_prefix}_{stage_name}.csv")
    try:
        df[columns_to_save].to_csv(file_path, index=False)
        logging.info(f"Saved {len(df)} rows to CSV: {file_path}")
    except Exception as e:
        logging.exception(f"Failed to save CSV {file_path}: {e}")


def is_futures_symbol(ticker):
    """Checks if a ticker represents a futures contract."""
    return ticker and ticker.startswith("/")


# --- Ticker Universe Generation ---
def get_market_metrics_robust(session, symbols, chunk_size=100):
    """Fetches market metrics in chunks, handling errors and tracking bad symbols."""
    bad_symbols_path = os.path.join(DATA_DIR, "bad_symbols.txt")
    known_bad_symbols = set(read_ticker_file(bad_symbols_path))
    symbols_to_query = [s for s in symbols if s not in known_bad_symbols]
    metrics_data = []
    newly_bad_symbols = set()

    for i in range(0, len(symbols_to_query), chunk_size):
        chunk = symbols_to_query[i : i + chunk_size]
        try:
            chunk_metrics = metrics.get_market_metrics(session, chunk)
            metrics_data.extend(
                {
                    "symbol": m.symbol,
                    "liquidity_rating": m.liquidity_rating,
                    "option_expiration_implied_volatilities": m.option_expiration_implied_volatilities,
                }
                for m in chunk_metrics
            )
        except Exception as e:
            logging.warning(f"Chunk failed, retrying individually: {e}")
            for symbol in chunk:
                try:
                    metric = metrics.get_market_metrics(session, [symbol])[0]
                    metrics_data.append(
                        {
                            "symbol": metric.symbol,
                            "liquidity_rating": metric.liquidity_rating,
                            "option_expiration_implied_volatilities": metric.option_expiration_implied_volatilities,
                        },
                    )
                except Exception:
                    newly_bad_symbols.add(symbol)

    if newly_bad_symbols:
        write_ticker_file(bad_symbols_path, known_bad_symbols | newly_bad_symbols)
        logging.info(f"Updated bad symbols: {len(newly_bad_symbols)} new, {len(known_bad_symbols | newly_bad_symbols)} total")

    return pd.DataFrame(metrics_data)


def _fetch_active_optionable_equities(session):
    """Fetches active optionable equities."""
    equities = instruments.Equity.get_active_equities(session)
    df = pd.DataFrame([{"symbol": e.symbol, "is_etf": e.is_etf} for e in equities])
    optionable_df = df[df["symbol"].isin([e.symbol for e in equities if e.option_tick_sizes])]
    logging.info(f"Found {len(optionable_df)} optionable equities")
    return optionable_df


def _get_metrics_and_filter_by_liquidity(session, optionable_df):
    """Fetches metrics and filters by liquidity thresholds."""
    if optionable_df.empty:
        logging.warning("No optionable equities provided")
        return pd.DataFrame()

    metrics_df = get_market_metrics_robust(session, optionable_df["symbol"].unique())
    if metrics_df.empty:
        logging.warning("No market metrics fetched")
        return pd.DataFrame()

    metrics_with_type = metrics_df[["symbol", "liquidity_rating", "option_expiration_implied_volatilities"]].merge(
        optionable_df[["symbol", "is_etf"]],
        on="symbol",
        how="left",
    )
    filtered_metrics = metrics_with_type[
        ((metrics_with_type["is_etf"] == False) & (metrics_with_type["liquidity_rating"] >= MIN_LIQUIDITY_NON_ETF))
        | ((metrics_with_type["is_etf"] == True) & (metrics_with_type["liquidity_rating"] >= MIN_LIQUIDITY_ETF))
    ]
    logging.info(f"Filtered to {len(filtered_metrics)} symbols by liquidity")
    return filtered_metrics[["symbol", "option_expiration_implied_volatilities"]]


def _filter_by_option_expirations_universe(metrics_df):
    """Filters symbols by option expiration count within a date range."""
    if metrics_df.empty or "option_expiration_implied_volatilities" not in metrics_df.columns:
        logging.warning("Metrics DataFrame empty or missing columns")
        return []

    today = pd.Timestamp.today().normalize()
    min_date = today + pd.Timedelta(days=MIN_EXPIRATION_DAYS_UNIVERSE)
    max_date = today + pd.Timedelta(days=MAX_EXPIRATION_DAYS_UNIVERSE)

    symbol_counts = {}
    for _, row in metrics_df.iterrows():
        expirations = [
            pd.to_datetime(opt.expiration_date, errors="coerce")
            for opt in row["option_expiration_implied_volatilities"] or []
            if opt and min_date < pd.to_datetime(opt.expiration_date, errors="coerce") < max_date
        ]
        symbol_counts[row["symbol"]] = len(set(exp for exp in expirations if pd.notna(exp)))

    valid_symbols = [symbol for symbol, count in symbol_counts.items() if count >= MIN_EXPIRATIONS_COUNT_UNIVERSE]
    result_df = pd.DataFrame({"symbol": valid_symbols, "num_valid_expirations_universe": [symbol_counts[symbol] for symbol in valid_symbols]})
    save_dataframe_to_csv(result_df, "ticker_universe", "expirations_filtered")
    logging.info(f"Found {len(valid_symbols)} symbols with sufficient expirations")
    return sorted(valid_symbols)


def get_ticker_universe_for_analysis(session):
    """Builds the ticker universe for analysis."""
    optionable_df = _fetch_active_optionable_equities(session)
    if optionable_df.empty:
        logging.error("No optionable equities found")
        return []

    filtered_metrics_df = _get_metrics_and_filter_by_liquidity(session, optionable_df)
    if filtered_metrics_df.empty:
        logging.error("No symbols passed liquidity filter")
        return []

    return _filter_by_option_expirations_universe(filtered_metrics_df)


# --- Option Chain Processing ---
EMPTY_TICKER_METRICS = {
    "avg_spread_percentage": float("nan"),
    "median_spread_percentage": float("nan"),
    "valid_options": float("nan"),
    "avg_options_per_expiration": 0,
    "per_expiration_data": {},
    "expirations_count": 0,
}


async def get_stock_prices_batch(session, tickers, retries=3, delay=5):
    """Fetches mid prices for a batch of tickers with retries."""
    for attempt in range(retries):
        try:
            data = await asyncio.get_event_loop().run_in_executor(None, lambda: get_market_data_by_type(session, equities=tickers))
            return {d.symbol: d.mid for d in data if d.mid is not None}
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1}/{retries} failed for stock prices: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay)
    logging.error(f"Failed to fetch stock prices after {retries} attempts")
    return {}


async def process_ticker_batch_parallel(session, tickers_batch, min_exp, max_exp, stock_prices):
    """Processes a batch of tickers concurrently with semaphore control."""
    semaphore = asyncio.Semaphore(MAX_WORKERS)

    async def bounded_process_ticker(ticker):
        async with semaphore:
            if is_futures_symbol(ticker):
                return {"entry": ticker, "stock_price": float("nan"), **EMPTY_TICKER_METRICS, "is_futures": True}
            return await process_ticker(session, ticker, min_exp, max_exp, stock_prices.get(ticker))

    tasks = [bounded_process_ticker(ticker) for ticker in tickers_batch]
    return await asyncio.gather(*tasks, return_exceptions=True)


async def _get_initial_option_data(session, ticker, min_exp, max_exp):
    """Fetches and filters option chain by expiration date range."""
    chain = await get_option_chain_async(session, ticker)
    filtered_chain = {k: v for k, v in chain.items() if isinstance(k, datetime.date) and min_exp <= k <= max_exp}
    if not filtered_chain:
        return None, None, None

    symbols_with_exp = [
        {"symbol": opt.symbol, "streamer_symbol": opt.streamer_symbol, "expiration_date": exp_date}
        for exp_date, options in filtered_chain.items()
        for opt in options
    ]
    streamer_symbols = list({item["streamer_symbol"] for item in symbols_with_exp})
    symbol_to_expiration = {item["streamer_symbol"]: item["expiration_date"] for item in symbols_with_exp}
    return filtered_chain, streamer_symbols, symbol_to_expiration


async def _fetch_and_filter_option_market_data(session, streamer_symbols):
    """Fetches and filters option market data by Delta and Open Interest."""
    if not streamer_symbols:
        return {}
    all_symbol_data = await fetch_market_data_efficient(session, streamer_symbols)
    return {
        symbol: data
        for symbol, data in all_symbol_data.items()
        if data["greeks"]
        and data["greeks"].delta is not None
        and Decimal("0") <= abs(data["greeks"].delta) <= DELTA_THRESHOLD
        and data["summary"]
        and getattr(data["summary"], "open_interest", 0) >= MIN_OPEN_INTEREST
    }


async def _stream_and_organize_spreads(session, filtered_data, stock_price, symbol_to_expiration):
    """Streams quotes and organizes spreads by expiration."""
    if not filtered_data or stock_price is None:
        return {}
    spreads = await stream_quotes_with_spread_efficient(session, filtered_data, stock_price)
    spreads_by_expiration = {}
    for symbol, spread_percentage in spreads.items():
        if spread_percentage is not None and not pd.isna(spread_percentage):
            exp_date = symbol_to_expiration.get(symbol)
            if exp_date:
                spreads_by_expiration.setdefault(exp_date, []).append(float(spread_percentage))
    return spreads_by_expiration


def _perform_final_calculations_and_filtering(ticker, stock_price, spreads_by_expiration):
    """Calculates metrics and applies tradability filters."""
    base_result = {"entry": ticker, "stock_price": stock_price, **EMPTY_TICKER_METRICS}
    if len(spreads_by_expiration) < MIN_EXPIRATIONS_COUNT_TRADABLE:
        base_result["expirations_count"] = len(spreads_by_expiration)
        return base_result

    per_expiration_data = {}
    total_valid_options = 0
    total_avg_spread = 0
    for exp_date, exp_spreads in spreads_by_expiration.items():
        if exp_spreads:
            avg_spread = statistics.mean(exp_spreads)
            median_spread = statistics.median(exp_spreads)
            option_count = len(exp_spreads)
            total_valid_options += option_count
            total_avg_spread += avg_spread
            per_expiration_data[exp_date.isoformat()] = {
                "avg_spread_percentage": avg_spread,
                "median_spread_percentage": median_spread,
                "valid_options": option_count,
            }

    expirations_count = len(per_expiration_data)
    if expirations_count == 0:
        base_result["expirations_count"] = len(spreads_by_expiration)
        return base_result

    avg_options_per_expiration = total_valid_options / expirations_count
    avg_of_avg_spreads = total_avg_spread / expirations_count
    all_spreads = [spread for spreads in spreads_by_expiration.values() for spread in spreads]
    median_spread = statistics.median(all_spreads) if all_spreads else float("nan")

    base_result.update(
        {
            "avg_spread_percentage": avg_of_avg_spreads,
            "median_spread_percentage": median_spread,
            "valid_options": avg_options_per_expiration,
            "expirations_count": expirations_count,
            "avg_options_per_expiration": avg_options_per_expiration,
            "per_expiration_data": per_expiration_data,
        },
    )

    if avg_options_per_expiration < MIN_VALID_OPTIONS or avg_of_avg_spreads >= MAX_AVG_SPREAD_PERCENTAGE:
        return base_result
    return base_result


async def process_ticker(session, ticker, min_exp, max_exp, stock_price):
    """Processes a single ticker through all option analysis stages."""
    result_template = {"entry": ticker, "stock_price": stock_price or float("nan"), **EMPTY_TICKER_METRICS}
    try:
        if stock_price is None or stock_price < MIN_STOCK_PRICE:
            logging.info(f"Skipping {ticker}: stock price {stock_price} < ${MIN_STOCK_PRICE}")
            return result_template

        chain_data, streamer_symbols, symbol_to_expiration = await _get_initial_option_data(session, ticker, min_exp, max_exp)
        if not streamer_symbols:
            logging.info(f"Skipping {ticker}: No options in range")
            result_template["expirations_count"] = 0
            return result_template

        filtered_option_details = await _fetch_and_filter_option_market_data(session, streamer_symbols)
        if not filtered_option_details:
            logging.info(f"Skipping {ticker}: No options passed filters")
            result_template["expirations_count"] = len(chain_data) if chain_data else 0
            return result_template

        spreads_by_expiration = await _stream_and_organize_spreads(session, filtered_option_details, stock_price, symbol_to_expiration)
        return _perform_final_calculations_and_filtering(ticker, stock_price, spreads_by_expiration)
    except Exception as e:
        logging.exception(f"Error processing {ticker}: {e}")
        return result_template


async def stream_quotes_with_spread_efficient(session, filtered_data, stock_price):
    """Streams quotes for options and calculates spreads."""
    symbols = list(filtered_data.keys())
    spreads = {}
    if not symbols or stock_price is None:
        return spreads

    batch_size = min(200, len(symbols))
    symbol_batches = [symbols[i : i + batch_size] for i in range(0, len(symbols), batch_size)]
    semaphore = asyncio.Semaphore(MAX_WORKERS)

    async def process_quote_batch(batch):
        async with semaphore:
            return await stream_quotes_batch(session, batch, stock_price)

    batch_results = await asyncio.gather(*[process_quote_batch(batch) for batch in symbol_batches], return_exceptions=True)
    for batch_spreads in batch_results:
        if isinstance(batch_spreads, dict):
            spreads.update(batch_spreads)
        elif isinstance(batch_spreads, Exception):
            logging.error(f"Batch failed in stream_quotes: {batch_spreads}")

    return spreads


async def stream_quotes_batch(session, symbols, stock_price):
    """Streams quotes for a batch of symbols and calculates spreads."""
    spreads = {}
    async with DXLinkStreamer(session) as streamer:
        await streamer.subscribe(Quote, symbols)
        pending_symbols = set(symbols)
        start_time = asyncio.get_event_loop().time()

        while pending_symbols and (asyncio.get_event_loop().time() - start_time < REQUEST_TIMEOUT):
            try:
                event = await asyncio.wait_for(streamer.listen(Quote).__anext__(), 0.1)
                symbol = event.event_symbol
                if symbol in pending_symbols:
                    if event.bid_price is not None and event.ask_price is not None and event.bid_price > 0:
                        spread = Decimal(str(event.ask_price)) - Decimal(str(event.bid_price))
                        spreads[symbol] = (spread / stock_price) * 100
                    else:
                        spreads[symbol] = None
                    pending_symbols.remove(symbol)
            except asyncio.TimeoutError:
                await asyncio.sleep(0.005)
            except Exception as e:
                logging.warning(f"Error in stream_quotes_batch: {e}")
                break

    for symbol in symbols:
        if symbol not in spreads:
            spreads[symbol] = None
    return spreads


# --- Watchlist Management ---
def extract_symbols_from_entries(entries):
    """Extracts symbols from watchlist entries."""
    return {entry.get("symbol") or getattr(entry, "symbol", None) for entry in entries if entry.get("symbol") or getattr(entry, "symbol", None)}


def get_futures_symbols(futures_entries):
    """Extracts futures symbols from entries."""
    return {symbol for symbol in extract_symbols_from_entries(futures_entries) if is_futures_symbol(symbol)}


async def update_watchlist(session, valid_tickers, futures_with_options):
    """Updates the private watchlist with filtered tickers and futures."""
    all_futures_symbols = get_futures_symbols(futures_with_options)
    ticker_removal_watch = read_ticker_file(TICKER_REMOVAL_FILE)
    ticker_add_watch = read_ticker_file(TICKER_ADD_FILE)

    # Symbols that are specifically forced (add or remove) and VIX
    special_managed_symbols = set(FORCED_TICKERS) | {VIX_SYMBOL}

    try:
        watchlists = PrivateWatchlist.get(session, WATCHLIST_NAME)
        current_symbols = extract_symbols_from_entries(watchlists.watchlist_entries)
    except Exception:
        logging.warning(f"Could not retrieve watchlist '{WATCHLIST_NAME}'")
        current_symbols = set()

    # Determine old managed symbols (not futures, not special)
    old_managed = [s for s in current_symbols if not is_futures_symbol(s) and s not in special_managed_symbols]

    # Determine valid managed symbols (not futures, not special, not force removed)
    valid_managed = []
    for t in valid_tickers:
        is_force_removed = False
        for remove_prefix in FORCED_REMOVE_TICKERS:
            if t.startswith(remove_prefix):  # Check for prefix removal for futures-like symbols
                is_force_removed = True
                break
        if not is_force_removed and not is_futures_symbol(t) and t not in special_managed_symbols:
            valid_managed.append(t)

    # Logic for removal watch and add watch files
    two_time_missing = [t for t in ticker_removal_watch if t not in valid_managed and t not in FORCED_TICKERS]
    new_removal_watch = [t for t in old_managed if t not in valid_managed and t not in FORCED_TICKERS]
    tickers_to_add = [t for t in valid_managed if t not in old_managed and t in ticker_add_watch]
    tickers_to_watch = [t for t in valid_managed if t not in old_managed and t not in ticker_add_watch]

    write_ticker_file(TICKER_REMOVAL_FILE, new_removal_watch)
    write_ticker_file(TICKER_ADD_FILE, tickers_to_watch)

    # Initialize final set with valid managed, tickers to add, and forced tickers
    final_symbols_set = set(valid_managed) | set(tickers_to_add) | set(FORCED_TICKERS)

    # Add futures symbols, excluding those starting with FORCED_REMOVE_TICKERS prefixes
    for f_symbol in all_futures_symbols:
        should_remove = False
        for remove_prefix in FORCED_REMOVE_TICKERS:
            if f_symbol.startswith(remove_prefix):
                should_remove = True
                break
        if not should_remove:
            final_symbols_set.add(f_symbol)

    # Ensure VIX_SYMBOL is included if it exists
    if VIX_SYMBOL and VIX_SYMBOL not in final_symbols_set:
        final_symbols_set.add(VIX_SYMBOL)

    # Convert the final set to a sorted list of dictionaries for the watchlist
    tickers_watchlist = [{"symbol": symbol} for symbol in sorted(list(final_symbols_set)) if symbol]

    try:
        PrivateWatchlist.remove(session, WATCHLIST_NAME)
    except Exception:
        logging.info(f"Watchlist '{WATCHLIST_NAME}' not found for removal")

    watchlist = PrivateWatchlist(name=WATCHLIST_NAME, watchlist_entries=tickers_watchlist, group_name="default", order_index=9999)
    watchlist.upload(session)

    logging.info(
        f"Updated watchlist '{WATCHLIST_NAME}' with {len(tickers_watchlist)} tickers. "
        f"Futures: {len(all_futures_symbols)}, Forced added: {FORCED_TICKERS}, Forced removed (prefixes): {FORCED_REMOVE_TICKERS}, "
        f"Removed (missed twice): {two_time_missing}, Removal watch: {new_removal_watch}, "
        f"Added: {tickers_to_add}, To watch: {tickers_to_watch}",
    )


async def main():
    """Main function to orchestrate ticker analysis and watchlist update."""
    initialize_directories_and_files()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    session = session_login()
    try:
        futures_with_options = PublicWatchlist.get(session, "Futures: With Options").watchlist_entries
        logging.info(f"Found {len(futures_with_options)} futures with options")
    except Exception:
        logging.exception("Error fetching futures watchlist")
        futures_with_options = []

    tickers_to_analyze = get_ticker_universe_for_analysis(session)
    if not tickers_to_analyze:
        logging.error("No tickers to analyze")
        return

    today = datetime.date.today()
    min_exp = today + datetime.timedelta(days=5)
    max_exp = today + datetime.timedelta(days=83)
    batch_size = max(10, MAX_WORKERS * 2)
    ticker_batches = [tickers_to_analyze[i : i + batch_size] for i in range(0, len(tickers_to_analyze), batch_size)]

    all_results = []
    with tqdm(total=len(tickers_to_analyze), desc="Processing tickers") as pbar:
        for batch in ticker_batches:
            stock_prices = await get_stock_prices_batch(session, batch)
            batch_results = await process_ticker_batch_parallel(session, batch, min_exp, max_exp, stock_prices)
            all_results.extend(r for r in batch_results if not isinstance(r, Exception))
            pbar.update(len(batch))

    results_df = (
        pd.DataFrame(all_results, columns=["entry", "stock_price", *EMPTY_TICKER_METRICS])
        if all_results
        else pd.DataFrame(columns=["entry", "stock_price", *EMPTY_TICKER_METRICS])
    )
    save_dataframe_to_csv(results_df, "tradable_tickers", "initial_processing_results")

    filtered_results_df = results_df[
        (results_df["stock_price"] >= MIN_STOCK_PRICE)
        & (results_df["valid_options"] >= MIN_VALID_OPTIONS)
        & (results_df["expirations_count"] >= MIN_EXPIRATIONS_COUNT_TRADABLE)
        & (results_df["avg_spread_percentage"] < MAX_AVG_SPREAD_PERCENTAGE)
        & (~results_df["avg_spread_percentage"].isna())
        & (~results_df["valid_options"].isna())
    ].sort_values("avg_spread_percentage")
    save_dataframe_to_csv(filtered_results_df, "tradable_tickers", "final_filtered_results")

    valid_tickers = filtered_results_df["entry"].tolist()
    logging.info(
        f"Analysis Summary: Total tickers: {len(tickers_to_analyze)}, "
        f"Processed: {len(results_df)}, Valid stock price: {len(results_df[results_df['stock_price'] >= MIN_STOCK_PRICE]) if 'stock_price' in results_df else 0}, "
        f"Valid after filters: {len(valid_tickers)}",
    )

    await update_watchlist(session, valid_tickers, futures_with_options)


if __name__ == "__main__":
    asyncio.run(main())
