import asyncio
import logging
import multiprocessing
from decimal import Decimal

import pandas as pd
from tastytrade import DXLinkStreamer
from tastytrade.dxfeed import Greeks, Quote, Summary
from tastytrade.instruments import Option, get_option_chain

# Constants moved from tradeable_tickers.py
BATCH_SIZE = 5000  # Used by fetch_market_data_efficient.
# Dynamically calculate MAX_WORKERS: twice the number of CPU cores.
MAX_WORKERS = multiprocessing.cpu_count() * 2
REQUEST_TIMEOUT = 3  # Used by fetch_market_data_batch and stream_quotes_batch (latter is still in tradeable_tickers)


async def get_option_chain_async(session, ticker):
    """Get option chain asynchronously."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: get_option_chain(session, ticker))


async def get_next_event(streamer, event_type):
    """Get next event with timeout."""
    try:
        event = await asyncio.wait_for(streamer.listen(event_type).__anext__(), 0.1)
        return (event, event_type)
    except (asyncio.TimeoutError, StopAsyncIteration):
        return None


async def fetch_market_data_batch(session, streamer_symbols):
    """Fetch market data for a batch of symbols, including Greeks, Summary, and Quotes."""
    symbol_data = {symbol: {"greeks": None, "summary": None, "quote": None} for symbol in streamer_symbols}

    if not streamer_symbols:
        return symbol_data

    async with DXLinkStreamer(session) as streamer:
        await streamer.subscribe(Greeks, streamer_symbols)
        await streamer.subscribe(Summary, streamer_symbols)
        await streamer.subscribe(Quote, streamer_symbols)  # Subscribe to Quotes

        # Sets to keep track of which symbols' data has been received for each type
        received_greeks = set()
        received_summary = set()
        received_quotes = set()  # New set for quotes
        start_time = asyncio.get_event_loop().time()

        try:
            while (
                len(received_greeks) < len(streamer_symbols)
                or len(received_summary) < len(streamer_symbols)
                or len(received_quotes) < len(streamer_symbols)
            ) and (asyncio.get_event_loop().time() - start_time < REQUEST_TIMEOUT):
                greek_task = asyncio.create_task(get_next_event(streamer, Greeks))
                summary_task = asyncio.create_task(get_next_event(streamer, Summary))
                quote_task = asyncio.create_task(get_next_event(streamer, Quote))  # New task for quotes

                done, pending = await asyncio.wait(
                    [greek_task, summary_task, quote_task],  # Include quote_task
                    timeout=0.1,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in pending:
                    task.cancel()

                for task in done:
                    try:
                        result = task.result()
                        if result:
                            event, event_type = result
                            symbol = event.event_symbol

                            if event_type == Greeks and symbol not in received_greeks:
                                symbol_data[symbol]["greeks"] = event
                                received_greeks.add(symbol)

                            elif event_type == Summary and symbol not in received_summary:
                                symbol_data[symbol]["summary"] = event
                                received_summary.add(symbol)

                            elif event_type == Quote and symbol not in received_quotes:  # Process Quote events
                                symbol_data[symbol]["quote"] = event
                                received_quotes.add(symbol)

                    except Exception:
                        logging.debug(
                            f"Error processing event in fetch_market_data_batch for {symbol if 'symbol' in locals() else 'unknown'}",
                            exc_info=True,
                        )
                        # Individual event processing error
        except Exception:
            logging.error("Error during fetch_market_data_batch event loop", exc_info=True)

    return symbol_data


async def fetch_market_data_efficient(session, streamer_symbols):
    """Efficiently fetch market data using batching."""
    if not streamer_symbols:
        return {}

    # Dynamic batch size for DXLink subscriptions, starting with a conservative limit
    # Adjust based on empirical performance or system constraints
    optimal_batch_size = min(BATCH_SIZE, 300)
    symbol_batches = [streamer_symbols[i : i + optimal_batch_size] for i in range(0, len(streamer_symbols), optimal_batch_size)]

    # Limit concurrent DXLinkStreamer instances or heavy subscription calls
    # Note: MAX_WORKERS here is from this shared_tasty_utils.py
    semaphore = asyncio.Semaphore(MAX_WORKERS)

    async def process_batch(batch, retries=2):
        async with semaphore:
            for attempt in range(retries):
                try:
                    return await fetch_market_data_batch(session, batch)
                except Exception as e:
                    logging.warning(f"Batch processing failed on attempt {attempt + 1}/{retries}: {e!s}")
                    if attempt == retries - 1:
                        logging.exception(f"Batch processing exhausted retries: {e!s}", exc_info=e)
                        raise
                    await asyncio.sleep(1)  # Wait before retry

    batch_results = await asyncio.gather(*[process_batch(batch) for batch in symbol_batches], return_exceptions=True)

    all_symbol_data = {}
    for batch_data in batch_results:
        if isinstance(batch_data, dict):
            all_symbol_data.update(batch_data)
        elif isinstance(batch_data, Exception):
            logging.error(f"A batch in fetch_market_data_efficient failed: {batch_data!s}", exc_info=batch_data)

    return all_symbol_data


async def fetch_option_data(session, underlying_symbols):
    """Fetches option chain data for a list of underlying symbols and
    maps option symbols to their streamer symbols.
    """
    symbol_to_streamer = {}
    for ticker in underlying_symbols:
        # Assuming get_option_chain_async is defined elsewhere or replaced with direct async call
        ticker_chain = await get_option_chain_async(session, ticker)
        if ticker_chain:  # Ensure ticker_chain is not None
            for options_list in ticker_chain.values():
                for option in options_list:
                    if isinstance(option, Option):  # Ensure it's an Option object
                        symbol_to_streamer[option.symbol] = option.streamer_symbol
    return symbol_to_streamer


async def fetch_greek_data(session, streamer_symbols):
    """Fetches market data, including greeks, for a list of option streamer symbols."""
    option_data = await fetch_market_data_efficient(session, streamer_symbols)
    rows = []
    for symbol, data in option_data.items():
        greeks = data.get("greeks")
        row = {"streamer_symbol": symbol}
        if greeks:
            row.update(
                {
                    "delta": Decimal(str(greeks.delta)),
                    "gamma": Decimal(str(greeks.gamma)),
                    "theta": Decimal(str(greeks.theta)),
                    "rho": Decimal(str(greeks.rho)),
                    "vega": Decimal(str(greeks.vega)),
                    "price": Decimal(str(greeks.price)),
                    "volatility": Decimal(str(greeks.volatility)),
                },
            )
        rows.append(row)
    return pd.DataFrame(rows, dtype=object)


def build_instrument_args(df):
    """Categorizes symbols from a DataFrame into instrument types for market data fetching."""
    args = {"indices": [], "cryptocurrencies": [], "equities": [], "futures": [], "future_options": [], "options": []}
    for _, row in df.iterrows():
        symbol = row["underlying_symbol"]
        instrument_type = row["instrument_type"]
        if instrument_type == "Equity":
            args["equities"].append(symbol)
        elif instrument_type == "Equity Option":
            args["options"].append(symbol)  # This should likely be row['symbol'] if it's an option symbol
        elif instrument_type == "Index":
            args["indices"].append(symbol)
        elif instrument_type == "Cryptocurrency":
            args["cryptocurrencies"].append(symbol)
        elif instrument_type == "Future":
            args["futures"].append(symbol)
        elif instrument_type == "Future Option":
            args["future_options"].append(symbol)  # This should likely be row['symbol']
    for k in args:
        args[k] = list(set(args[k]))  # Ensure unique symbols
    return args


def flatten_market_metric_info(obj):
    """Flattens a market metric object from the Tastytrade API."""
    d = obj.__dict__.copy()
    if "earnings" in d and d["earnings"] is not None:
        d.update({f"earnings_{k}": v for k, v in d["earnings"].__dict__.items()})
        del d["earnings"]
    if "option_expiration_implied_volatilities" in d and d["option_expiration_implied_volatilities"] is not None:
        for i, exp in enumerate(d["option_expiration_implied_volatilities"][:3]):
            if hasattr(exp, "__dict__"):  # Check if exp is an object with __dict__
                for k, v in exp.__dict__.items():
                    d[f"option_exp_{i + 1}_{k}"] = v
        del d["option_expiration_implied_volatilities"]
    return d
