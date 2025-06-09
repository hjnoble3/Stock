import nest_asyncio
import numpy as np
import pandas as pd
from chain import getFilteredOptionData
from scipy.stats import norm
from session_login import session_login

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


def calculate_probability_of_profit(current_price, breakeven, standard_deviation, option_type):
    """Calculates the probability of profit for various option strategies assuming
    the underlying price follows a normal distribution.
    """
    if pd.isna(standard_deviation) or standard_deviation == 0:
        return 0.0

    current_price = float(current_price)
    standard_deviation = float(standard_deviation)

    if isinstance(breakeven, tuple):
        # For strategies with two breakeven points (e.g., Short Strangle, Iron Condor, Broken Wing Butterfly)
        lower_be, upper_be = (
            float(breakeven[0]) if not pd.isna(breakeven[0]) else np.nan,
            float(breakeven[1]) if not pd.isna(breakeven[1]) else np.nan,
        )
        if option_type in ("Short Strangle", "Iron Condor", "Broken Wing Butterfly"):
            # Probability of being *within* the breakeven range
            return norm.cdf((upper_be - current_price) / standard_deviation) - norm.cdf((lower_be - current_price) / standard_deviation)
    else:
        # For strategies with a single breakeven point
        breakeven = float(breakeven) if not pd.isna(breakeven) else np.nan
        if option_type in ("Long Call", "Short Put"):
            # Probability of price > breakeven
            z_score = (breakeven - current_price) / standard_deviation
            return 1 - norm.cdf(z_score)
        if option_type in ("Long Put", "Short Call"):
            # Probability of price < breakeven
            z_score = (breakeven - current_price) / standard_deviation
            return norm.cdf(z_score)
    return 0.0


def expected_tail_value(mean, std_dev, percentile_for_VaR, tail_type):
    """Calculates the expected value of a normal distribution in its tail (CVaR price).
    This represents the average value of the underlying, given that it falls
    into a certain extreme percentile.

    Args:
        mean (float): The mean of the normal distribution (e.g., current_price).
        std_dev (float): The standard deviation of the normal distribution.
        percentile_for_VaR (float): The percentile that defines the VaR threshold.
                                    For a 95% CVaR (worst 5% outcomes), this would be:
                                    - 0.95 for the upper tail (e.g., for short calls)
                                    - 0.05 for the lower tail (e.g., for short puts)
        tail_type (str): 'upper' for upper tail, 'lower' for lower tail.

    Returns:
        float: The expected value in the specified tail. Returns mean if std_dev is 0 or invalid.

    """
    if pd.isna(std_dev) or std_dev <= 0:
        return mean

    # Z-score corresponding to the given percentile for VaR
    z_score_at_VaR = norm.ppf(percentile_for_VaR)

    if tail_type == "upper":
        # Formula for Expected Shortfall (CVaR) price for upper tail:
        # E[X | X > VaR_percentile] = mu + sigma * (phi(z_percentile) / (1 - Phi(z_percentile)))
        # where phi is PDF and Phi is CDF of the standard normal distribution.
        inverse_mills_ratio = norm.pdf(z_score_at_VaR) / (1 - norm.cdf(z_score_at_VaR))
        return mean + std_dev * inverse_mills_ratio
    if tail_type == "lower":
        # Formula for Expected Shortfall (CVaR) price for lower tail:
        # E[X | X < VaR_percentile] = mu - sigma * (phi(z_percentile) / Phi(z_percentile))
        # Note: percentile_for_VaR here should be for the lower tail (e.g., 0.05 for 95% CVaR)
        # z_score_at_VaR will be negative for percentiles < 0.5.
        inverse_mills_ratio = norm.pdf(z_score_at_VaR) / norm.cdf(z_score_at_VaR)
        return mean - std_dev * inverse_mills_ratio
    raise ValueError("tail_type must be 'upper' or 'lower'")


def calculate_cvar(
    strategy_type,
    mid_price,  # This is the net premium received for short options, or paid for long options/debit spreads
    notional_multiplier,
    strike_leg1=None,
    strike_leg2=None,
    pop=None,  # Probability of Profit, kept for consistency but not used in standard CVaR calculation
    current_price=None,
    standard_deviation=None,
):
    """Calculate Conditional Value at Risk (CVaR) for various option strategies.
    For defined-risk strategies, CVaR is their maximum potential loss.
    For undefined-risk strategies (short options, strangles), it's the expected
    loss in the specified tail of the distribution.
    """
    mid_price = float(mid_price)
    notional_multiplier = float(notional_multiplier)

    # Long positions: CVaR is premium paid (which is the maximum loss)
    if strategy_type in ("Long Call", "Long Put", "Long Strangle"):
        return abs(mid_price) * notional_multiplier

    # Credit Spreads: CVaR is max loss (spread width minus net credit received)
    if "Credit Spread" in strategy_type:
        if strike_leg1 is None or strike_leg2 is None:
            return np.nan
        spread_width = abs(float(strike_leg1) - float(strike_leg2))
        max_loss = (spread_width - mid_price) * notional_multiplier
        return max(max_loss, 0)  # CVaR is the max loss for credit spreads (cannot be negative)

    # Debit Spreads: Max loss is debit paid
    if "Debit Spread" in strategy_type:
        # For Debit Spreads, max loss is the premium paid (mid_price is net debit)
        max_loss = abs(mid_price) * notional_multiplier
        return max_loss

    # For other strategies (Short Put, Short Call, Short Strangle, Iron Condor, Broken Wing Butterfly),
    # compute expected tail loss based on a normal distribution assumption.
    if current_price is None or standard_deviation is None:
        return np.nan

    current_price = float(current_price)
    standard_deviation = float(standard_deviation)

    # Short Put: CVaR is the expected loss in the lower tail of the distribution.
    if strategy_type == "Short Put":
        if strike_leg1 is None:
            return np.nan
        # Determine the percentile for VaR for the lower tail (e.g., 0.05 for 95% CVaR)
        percentile_for_cvar_tail = 1 - CONFIDENCE_LEVEL
        expected_tail_price = expected_tail_value(current_price, standard_deviation, percentile_for_cvar_tail, "lower")

        # Loss at this expected tail price: intrinsic value of the put - premium received
        # Ensure loss is non-negative if the price goes below strike_leg1
        loss_at_tail_price = max(0, float(strike_leg1) - expected_tail_price)

        # CVaR is this expected intrinsic loss, offset by the premium received for the short put
        cvar_value = (loss_at_tail_price - mid_price) * notional_multiplier
        return max(0, cvar_value)  # CVaR represents a loss, so it should be non-negative

    # Short Call: CVaR is the expected loss in the upper tail of the distribution.
    if strategy_type == "Short Call":
        if strike_leg1 is None:
            return np.nan
        # Determine the percentile for VaR for the upper tail (e.g., 0.95 for 95% CVaR)
        percentile_for_cvar_tail = CONFIDENCE_LEVEL
        expected_tail_price = expected_tail_value(current_price, standard_deviation, percentile_for_cvar_tail, "upper")

        # Loss at this expected tail price: intrinsic value of the call - premium received
        # Ensure loss is non-negative if the price goes above strike_leg1
        loss_at_tail_price = max(0, expected_tail_price - float(strike_leg1))

        # CVaR is this expected intrinsic loss, offset by the premium received for the short call
        cvar_value = (loss_at_tail_price - mid_price) * notional_multiplier
        return max(0, cvar_value)

    # Short Strangle: CVaR is the maximum of the expected losses from either the put side or the call side.
    if strategy_type == "Short Strangle":
        if strike_leg1 is None or strike_leg2 is None:
            return np.nan  # strike_leg1 is put strike, strike_leg2 is call strike

        # Calculate expected price in the lower tail (relevant for the short put leg)
        expected_price_lower_tail = expected_tail_value(current_price, standard_deviation, 1 - CONFIDENCE_LEVEL, "lower")
        # Calculate expected price in the upper tail (relevant for the short call leg)
        expected_price_upper_tail = expected_tail_value(current_price, standard_deviation, CONFIDENCE_LEVEL, "upper")

        # Calculate the potential loss from the put side if the price drops into the lower tail
        # This is the intrinsic value of the put at the expected lower tail price
        loss_from_put_side = max(0, float(strike_leg1) - expected_price_lower_tail)

        # Calculate the potential loss from the call side if the price rises into the upper tail
        # This is the intrinsic value of the call at the expected upper tail price
        # FIXED: Changed 'expected_tail_price' to 'expected_price_upper_tail' to resolve UnboundLocalError
        loss_from_call_side = max(0, expected_price_upper_tail - float(strike_leg2))

        # The total expected intrinsic loss in the tail is the maximum of the two individual leg losses.
        # This is a simplification but captures the dominant tail risk of the strangle.
        total_intrinsic_loss_in_tail = max(loss_from_put_side, loss_from_call_side)

        # CVaR is this maximum intrinsic loss, offset by the total premium received for the strangle.
        cvar_value = (total_intrinsic_loss_in_tail - mid_price) * notional_multiplier
        return max(0, cvar_value)  # Ensure CVaR is non-negative

    # Iron Condor and Broken Wing Butterfly: CVaR is the defined maximum loss.
    # We're passing the calculated max_loss (per contract) as mid_price
    # and using "Long Call" strategy type to simply return abs(mid_price) * notional_multiplier
    if strategy_type in ("Iron Condor", "Broken Wing Butterfly", "Long Calendar"):
        # For these defined-risk strategies, max loss is typically passed as 'mid_price' to calculate_cvar
        return abs(mid_price) * notional_multiplier

    # Short Calendar and Short Diagonal: These are typically unlimited risk strategies
    if strategy_type in ("Short Calendar", "Short Diagonal"):
        return np.inf  # Reflects potentially unlimited risk

    # Long Diagonal: Max loss is initial debit paid (defined risk)
    if strategy_type == "Long Diagonal":
        return abs(mid_price) * notional_multiplier

    return np.nan  # Return NaN if strategy type is not handled


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


def create_spreads(grouped, current_price):
    """Identifies and creates various option spread strategies (credit, debit, ratio backspreads)."""
    spreads = []
    current_price = float(current_price)

    for (exp_date, opt_type), group in grouped:
        long_opts = group[group["position_type"].str.contains("Long")].sort_values("strike_price")
        short_opts = group[group["position_type"].str.contains("Short")].sort_values("strike_price")
        if long_opts.empty or short_opts.empty:
            continue

        notional_multiplier = float(long_opts["notional_multiplier"].iloc[0]) if not long_opts.empty else 1.0
        standard_deviation = float(long_opts["standard_deviation"].iloc[0]) if not long_opts.empty else np.nan

        for _, short_opt in short_opts.iterrows():
            short_strike = short_opt["strike_price"]
            for _, long_opt in long_opts.iterrows():
                long_strike = long_opt["strike_price"]

                # Call Credit Spread: Sell OTM Call, Buy further OTM Call (short strike < long strike, both > current_price)
                if opt_type == "C" and short_strike < long_strike and short_strike > current_price:
                    net_credit = short_opt["mid_price"] - long_opt["mid_price"]
                    spread_width = abs(long_strike - short_strike)
                    max_profit = net_credit * notional_multiplier
                    if max_profit < (spread_width / 4.0) * notional_multiplier:  # Filter for minimum profit
                        continue
                    lower_breakeven, upper_breakeven = np.nan, short_strike + net_credit
                    # Probability of profit is for the short leg or the spread itself
                    pop = calculate_probability_of_profit(current_price, upper_breakeven, standard_deviation, "Short Call")
                    cvar = calculate_cvar(
                        "Credit Spread",
                        net_credit,
                        notional_multiplier,
                        short_strike,  # Strike of the short leg
                        long_strike,  # Strike of the long leg
                        pop,
                        current_price,
                        standard_deviation,
                    )
                    spreads.append(
                        {
                            "expiration_date": exp_date,
                            "option_type": opt_type,
                            "strategy": "Call Credit Spread",
                            "strikes": f"{short_strike},{long_strike}",
                            "mid_price": net_credit,
                            "max_profit": max_profit,
                            "delta": short_opt["delta"] + long_opt["delta"],
                            "theta": short_opt["theta"] + long_opt["theta"],
                            "beta_weighted_delta": short_opt["beta_weighted_delta"] + long_opt["beta_weighted_delta"],
                            "position_type": f"Short Call {short_strike}/Long Call {long_strike}",
                            "lower_breakeven": lower_breakeven,
                            "upper_breakeven": upper_breakeven,
                            "probability_of_profit": pop,
                            "cvar": cvar,
                        },
                    )

                # Call Debit Spread: Buy ITM Call, Sell further ITM Call (long strike < short strike, both < current_price)
                elif opt_type == "C" and long_strike < short_strike and long_strike < current_price:
                    net_debit = long_opt["mid_price"] - short_opt["mid_price"]
                    spread_width = abs(long_strike - short_strike)
                    max_profit = (spread_width - net_debit) * notional_multiplier
                    if max_profit < (spread_width / 4.0) * notional_multiplier:  # Filter for minimum profit
                        continue
                    lower_breakeven, upper_breakeven = long_strike + net_debit, np.nan
                    # Probability of profit is for the long leg or the spread itself
                    pop = calculate_probability_of_profit(current_price, lower_breakeven, standard_deviation, "Long Call")
                    cvar = calculate_cvar(
                        "Debit Spread",
                        net_debit,
                        notional_multiplier,
                        long_strike,  # Strike of the long leg
                        short_strike,  # Strike of the short leg
                        pop,
                        current_price,
                        standard_deviation,
                    )
                    spreads.append(
                        {
                            "expiration_date": exp_date,
                            "option_type": opt_type,
                            "strategy": "Call Debit Spread",
                            "strikes": f"{long_strike},{short_strike}",
                            "mid_price": -net_debit,  # Store as negative to indicate debit
                            "max_profit": max_profit,
                            "delta": long_opt["delta"] + short_opt["delta"],
                            "theta": long_opt["theta"] + short_opt["theta"],
                            "beta_weighted_delta": long_opt["beta_weighted_delta"] + short_opt["beta_weighted_delta"],
                            "position_type": f"Long Call {long_strike}/Short Call {short_strike}",
                            "lower_breakeven": lower_breakeven,
                            "upper_breakeven": upper_breakeven,
                            "probability_of_profit": pop,
                            "cvar": cvar,
                        },
                    )

                # Put Credit Spread: Sell OTM Put, Buy further OTM Put (long strike < short strike, both < current_price)
                elif opt_type == "P" and long_strike < short_strike < current_price:
                    net_credit = short_opt["mid_price"] - long_opt["mid_price"]
                    spread_width = abs(long_strike - short_strike)
                    max_profit = net_credit * notional_multiplier
                    if max_profit < (spread_width / 4.0) * notional_multiplier:  # Filter for minimum profit
                        continue
                    lower_breakeven, upper_breakeven = short_strike - net_credit, np.nan
                    # Probability of profit is for the short leg or the spread itself
                    pop = calculate_probability_of_profit(current_price, lower_breakeven, standard_deviation, "Short Put")
                    cvar = calculate_cvar(
                        "Credit Spread",
                        net_credit,
                        notional_multiplier,
                        short_strike,  # Strike of the short leg
                        long_strike,  # Strike of the long leg
                        pop,
                        current_price,
                        standard_deviation,
                    )
                    spreads.append(
                        {
                            "expiration_date": exp_date,
                            "option_type": opt_type,
                            "strategy": "Put Credit Spread",
                            "strikes": f"{short_strike},{long_strike}",
                            "mid_price": net_credit,
                            "max_profit": max_profit,
                            "delta": short_opt["delta"] + long_opt["delta"],
                            "theta": short_opt["theta"] + long_opt["theta"],
                            "beta_weighted_delta": short_opt["beta_weighted_delta"] + long_opt["beta_weighted_delta"],
                            "position_type": f"Short Put {short_strike}/Long Put {long_strike}",
                            "lower_breakeven": lower_breakeven,
                            "upper_breakeven": upper_breakeven,
                            "probability_of_profit": pop,
                            "cvar": cvar,
                        },
                    )
                # Put Debit Spread: Buy ITM Put, Sell further ITM Put (short strike < long strike, both > current_price)
                elif opt_type == "P" and short_strike < long_strike and long_strike > current_price:
                    net_debit = long_opt["mid_price"] - short_opt["mid_price"]
                    spread_width = abs(long_strike - short_strike)
                    max_profit = (spread_width - net_debit) * notional_multiplier
                    if max_profit < (spread_width / 4.0) * notional_multiplier:  # Filter for minimum profit
                        continue
                    lower_breakeven, upper_breakeven = np.nan, long_strike - net_debit
                    # Probability of profit is for the long leg or the spread itself
                    pop = calculate_probability_of_profit(current_price, upper_breakeven, standard_deviation, "Long Put")
                    cvar = calculate_cvar(
                        "Debit Spread",
                        net_debit,
                        notional_multiplier,
                        long_strike,  # Strike of the long leg
                        short_strike,  # Strike of the short leg
                        pop,
                        current_price,
                        standard_deviation,
                    )
                    spreads.append(
                        {
                            "expiration_date": exp_date,
                            "option_type": opt_type,
                            "strategy": "Put Debit Spread",
                            "strikes": f"{long_strike},{short_strike}",
                            "mid_price": -net_debit,  # Store as negative to indicate debit
                            "max_profit": max_profit,
                            "delta": long_opt["delta"] + short_opt["delta"],
                            "theta": long_opt["theta"] + short_opt["theta"],
                            "beta_weighted_delta": long_opt["beta_weighted_delta"] + short_opt["beta_weighted_delta"],
                            "position_type": f"Long Put {long_strike}/Short Put {short_strike}",
                            "lower_breakeven": lower_breakeven,
                            "upper_breakeven": upper_breakeven,
                            "probability_of_profit": pop,
                            "cvar": cvar,
                        },
                    )
                # Call Ratio Backspread (Long 1 Call, Short 2 Calls)
                elif opt_type == "C" and long_strike < short_strike and long_strike > current_price:
                    net_credit_or_debit = (2 * short_opt["mid_price"]) - long_opt["mid_price"]
                    spread_width = short_strike - long_strike
                    # Max loss if price hits short strike (from the 1x2 spread portion)
                    max_loss_at_short_strike_per_contract = spread_width - (short_opt["mid_price"] - long_opt["mid_price"])

                    # Omnidirectional condition: extra short leg's credit > max loss of the narrower spread
                    # The "spread" here refers to the (Long A, Short B) part of the ratio spread.
                    # The max loss of this spread is (B-A) - (P_B - P_A)
                    # The "extra short leg's credit" is P_B (short_opt["mid_price"])
                    # Re-evaluating: max_loss_at_short_strike_per_contract is essentially (width of spread) - (net credit of 1x1 spread)
                    # The condition is: short_opt["mid_price"] > max_loss_of_that_1x1_spread.
                    # max_loss_of_that_1x1_spread is (short_strike - long_strike) - (short_opt["mid_price"] - long_opt["mid_price"])

                    # Correct interpretation of "extra short leg's credit is more than the max loss of the spread":
                    # This implies P_B (credit from one short B) > (B-A) - (P_B - P_A)
                    # Which simplifies to: 2 * P_B > (B-A) + P_A
                    # This means 2 * short_opt["mid_price"] > (short_strike - long_strike) + long_opt["mid_price"]
                    if not ((2 * short_opt["mid_price"]) > (short_strike - long_strike) + long_opt["mid_price"]):
                        continue

                    # If net_credit_or_debit <= 0, it's a debit ratio spread, which might not be omnidirectional
                    if net_credit_or_debit <= 0:
                        continue

                    max_profit_ratio = np.inf  # Theoretically unlimited profit on one side
                    upper_breakeven = short_strike + spread_width - net_credit_or_debit
                    lower_breakeven = np.nan  # Can be non-existent or complex to calculate
                    pop = calculate_probability_of_profit(
                        current_price,
                        upper_breakeven,
                        standard_deviation,
                        "Long Call",
                    )  # Using Long Call for POP calculation as it's a one-sided breakeven
                    cvar = calculate_cvar(
                        "Long Call",  # Categorized as Long Call for CVaR, since it's theoretically unlimited profit on one side
                        max_loss_at_short_strike_per_contract,  # The max loss at the short strike is passed as 'mid_price' to CVaR
                        notional_multiplier,
                        pop=pop,
                        current_price=current_price,
                        standard_deviation=standard_deviation,
                    )
                    spreads.append(
                        {
                            "expiration_date": exp_date,
                            "option_type": opt_type,
                            "strategy": "Call Ratio Backspread (Omnidirectional)",
                            "strikes": f"{long_strike},{short_strike}",
                            "mid_price": net_credit_or_debit,
                            "max_profit": max_profit_ratio,
                            "delta": long_opt["delta"] - (2 * short_opt["delta"]),
                            "theta": long_opt["theta"] - (2 * short_opt["theta"]),
                            "beta_weighted_delta": long_opt["beta_weighted_delta"] - (2 * short_opt["beta_weighted_delta"]),
                            "position_type": f"Long Call {long_strike}/2x Short Call {short_strike}",
                            "lower_breakeven": lower_breakeven,
                            "upper_breakeven": upper_breakeven,
                            "probability_of_profit": pop,
                            "cvar": cvar,
                        },
                    )
                # Put Ratio Backspread
                elif opt_type == "P" and long_strike > short_strike and long_strike < current_price:
                    net_credit_or_debit = (2 * short_opt["mid_price"]) - long_opt["mid_price"]
                    spread_width = long_strike - short_strike
                    # Max loss if price hits short strike (from the 1x2 spread portion)
                    max_loss_at_short_strike_per_contract = spread_width - (short_opt["mid_price"] - long_opt["mid_price"])

                    # Omnidirectional condition: extra short leg's credit > max loss of the narrower spread
                    # Correct interpretation: 2 * P_B > (C-B) + P_C
                    # This means 2 * short_opt["mid_price"] > (long_strike - short_strike) + long_opt["mid_price"]
                    if not ((2 * short_opt["mid_price"]) > (long_strike - short_strike) + long_opt["mid_price"]):
                        continue

                    # If net_credit_or_debit <= 0, it's a debit ratio spread, which might not be omnidirectional
                    if net_credit_or_debit <= 0:
                        continue

                    max_profit_ratio = np.inf  # Theoretically unlimited profit on one side
                    lower_breakeven = short_strike - spread_width + net_credit_or_debit
                    upper_breakeven = np.nan  # Can be non-existent or complex to calculate
                    pop = calculate_probability_of_profit(
                        current_price,
                        lower_breakeven,
                        standard_deviation,
                        "Long Put",
                    )  # Using Long Put for POP calculation
                    cvar = calculate_cvar(
                        "Long Put",  # Categorized as Long Put for CVaR, since it's theoretically unlimited profit on one side
                        max_loss_at_short_strike_per_contract,  # The max loss at the short strike is passed as 'mid_price' to CVaR
                        notional_multiplier,
                        pop=pop,
                        current_price=current_price,
                        standard_deviation=standard_deviation,
                    )
                    spreads.append(
                        {
                            "expiration_date": exp_date,
                            "option_type": opt_type,
                            "strategy": "Put Ratio Backspread (Omnidirectional)",
                            "strikes": f"{long_strike},{short_strike}",
                            "mid_price": net_credit_or_debit,
                            "max_profit": max_profit_ratio,
                            "delta": long_opt["delta"] - (2 * short_opt["delta"]),
                            "theta": long_opt["theta"] - (2 * short_opt["theta"]),
                            "beta_weighted_delta": long_opt["beta_weighted_delta"] - (2 * short_opt["beta_weighted_delta"]),
                            "position_type": f"Long Put {long_strike}/2x Short Put {short_strike}",
                            "lower_breakeven": lower_breakeven,
                            "upper_breakeven": upper_breakeven,
                            "probability_of_profit": pop,
                            "cvar": cvar,
                        },
                    )
    return spreads


def create_strangles(exp_groups, current_price):
    """Identifies and creates short strangle strategies."""
    strangles = []
    current_price = float(current_price)

    for exp_date, group in exp_groups:
        # Filter for short calls and short puts within the current expiration group
        short_calls = group[(group["option_type"] == "C") & (group["position_type"].str.contains("Short"))].sort_values("strike_price")
        short_puts = group[(group["option_type"] == "P") & (group["position_type"].str.contains("Short"))].sort_values("strike_price")

        if short_calls.empty or short_puts.empty:
            continue

        notional_multiplier = float(group["notional_multiplier"].iloc[0]) if not group.empty else 1.0
        standard_deviation = float(group["standard_deviation"].iloc[0]) if not group.empty else np.nan

        # Filter calls above current price and puts below current price for OTM strangles
        short_calls = short_calls[short_calls["strike_price"] > current_price]
        short_puts = short_puts[short_puts["strike_price"] < current_price]

        for _, short_put in short_puts.iterrows():
            for _, short_call in short_calls.iterrows():
                short_put_strike = short_put["strike_price"]
                short_call_strike = short_call["strike_price"]

                # Ensure call strike is higher than put strike for a valid strangle
                if short_call_strike <= short_put_strike:
                    continue

                net_credit = short_put["mid_price"] + short_call["mid_price"]
                max_profit = net_credit * notional_multiplier
                lower_breakeven = short_put_strike - net_credit
                upper_breakeven = short_call_strike + net_credit

                # Calculate probability of profit for the strangle
                pop = calculate_probability_of_profit(current_price, (lower_breakeven, upper_breakeven), standard_deviation, "Short Strangle")

                # Filter strangles based on a minimum probability of profit
                if pop < 0.60:
                    continue

                # Calculate CVaR for the short strangle
                cvar = calculate_cvar(
                    "Short Strangle",
                    net_credit,  # Total net credit for the strangle
                    notional_multiplier,
                    short_put_strike,  # strike_leg1 is the put strike
                    short_call_strike,  # strike_leg2 is the call strike
                    pop,
                    current_price,
                    standard_deviation,
                )
                strangles.append(
                    {
                        "expiration_date": exp_date,
                        "strategy": "Short Strangle",
                        "strikes": f"{short_put_strike},{short_call_strike}",
                        "mid_price": net_credit,
                        "max_profit": max_profit,
                        "delta": short_put["delta"] + short_call["delta"],
                        "theta": short_put["theta"] + short_call["theta"],
                        "beta_weighted_delta": short_put["beta_weighted_delta"] + short_call["beta_weighted_delta"],
                        "position_type": f"Short Put {short_put_strike}/Short Call {short_call_strike}",
                        "lower_breakeven": lower_breakeven,
                        "upper_breakeven": upper_breakeven,
                        "probability_of_profit": pop,
                        "cvar": cvar,
                    },
                )
    return strangles


def create_iron_condors(exp_groups, current_price):
    """Identifies and creates Iron Condor strategies.
    An Iron Condor combines a bear call spread and a bull put spread.
    """
    iron_condors = []
    current_price = float(current_price)

    for exp_date, group in exp_groups:
        # Get all long and short calls/puts for this expiration
        long_puts = group[(group["option_type"] == "P") & (group["position_type"].str.contains("Long"))].sort_values("strike_price")
        short_puts = group[(group["option_type"] == "P") & (group["position_type"].str.contains("Short"))].sort_values("strike_price")
        long_calls = group[(group["option_type"] == "C") & (group["position_type"].str.contains("Long"))].sort_values("strike_price")
        short_calls = group[(group["option_type"] == "C") & (group["position_type"].str.contains("Short"))].sort_values("strike_price")

        # Filter for OTM short legs for a classic Iron Condor setup
        short_puts_otm = short_puts[short_puts["strike_price"] < current_price]
        short_calls_otm = short_calls[short_calls["strike_price"] > current_price]

        # Ensure we have enough options to form a condor
        if short_puts_otm.empty or short_calls_otm.empty or long_puts.empty or long_calls.empty:
            continue

        notional_multiplier = float(group["notional_multiplier"].iloc[0]) if not group.empty else 1.0
        standard_deviation = float(group["standard_deviation"].iloc[0]) if not group.empty else np.nan

        # Iterate through combinations to form condors
        for _, sp in short_puts_otm.iterrows():  # sp = short put
            for _, lp in long_puts.iterrows():  # lp = long put
                # Condition for Bull Put Spread: Long Put strike < Short Put strike
                if not (lp["strike_price"] < sp["strike_price"]):
                    continue

                for _, sc in short_calls_otm.iterrows():  # sc = short call
                    for _, lc in long_calls.iterrows():  # lc = long call
                        # Condition for Bear Call Spread: Short Call strike < Long Call strike
                        if not (lc["strike_price"] > sc["strike_price"]):
                            continue

                        # Ensure the put and call spreads do not overlap and maintain order
                        # lp_strike < sp_strike < sc_strike < lc_strike
                        if not (sp["strike_price"] < sc["strike_price"]):
                            continue
                        if not (lp["strike_price"] < sp["strike_price"]):
                            continue  # Redundant check but ensures clarity in logic flow
                        if not (sc["strike_price"] < lc["strike_price"]):
                            continue  # Redundant check

                        # Calculate net credit for the entire Iron Condor
                        net_credit = (sp["mid_price"] + sc["mid_price"]) - (lp["mid_price"] + lc["mid_price"])

                        # Only proceed if there's a net credit received
                        if net_credit <= 0:
                            continue

                        # Max Profit for an Iron Condor is the net credit received
                        max_profit = net_credit * notional_multiplier

                        # Max Loss for an Iron Condor:
                        # This is the difference between the strikes of one of the spreads
                        # minus the net credit received for the entire condor.
                        # Assuming call_spread_width == put_spread_width for a symmetrical condor.
                        call_spread_width = lc["strike_price"] - sc["strike_price"]
                        # put_spread_width = sp["strike_price"] - lp["strike_price"] # For validation

                        # The max loss per contract is the width of the vertical spread minus the net credit.
                        max_loss_per_contract = call_spread_width - net_credit

                        # Max loss must be positive if we received a credit
                        if max_loss_per_contract <= 0:
                            continue

                        max_loss_total = max_loss_per_contract * notional_multiplier

                        # Breakeven points for Iron Condor
                        lower_breakeven = sp["strike_price"] - net_credit
                        upper_breakeven = sc["strike_price"] + net_credit

                        # Probability of Profit: price staying between lower_breakeven and upper_breakeven
                        pop = calculate_probability_of_profit(current_price, (lower_breakeven, upper_breakeven), standard_deviation, "Iron Condor")

                        # Filter for a minimum probability of profit
                        if pop < 0.60:
                            continue

                        cvar = calculate_cvar(
                            "Iron Condor",  # Use new strategy type for explicit handling in calculate_cvar
                            max_loss_per_contract,  # Pass max_loss per contract to calculate_cvar
                            notional_multiplier,
                            pop=pop,
                            current_price=current_price,
                            standard_deviation=standard_deviation,
                        )

                        iron_condors.append(
                            {
                                "expiration_date": exp_date,
                                "strategy": "Iron Condor",
                                "strikes": f"{lp['strike_price']},{sp['strike_price']},{sc['strike_price']},{lc['strike_price']}",
                                "mid_price": net_credit,
                                "max_profit": max_profit,
                                "max_loss": max_loss_total,  # Explicitly add max_loss to the dictionary
                                "delta": lp["delta"] + sp["delta"] + sc["delta"] + lc["delta"],
                                "theta": lp["theta"] + sp["theta"] + sc["theta"] + lc["theta"],
                                "beta_weighted_delta": lp["beta_weighted_delta"]
                                + sp["beta_weighted_delta"]
                                + sc["beta_weighted_delta"]
                                + lc["beta_weighted_delta"],
                                "position_type": f"Long Put {lp['strike_price']}/Short Put {sp['strike_price']}/Short Call {sc['strike_price']}/Long Call {lc['strike_price']}",
                                "lower_breakeven": lower_breakeven,
                                "upper_breakeven": upper_breakeven,
                                "probability_of_profit": pop,
                                "cvar": cvar,
                            },
                        )
    return iron_condors


def create_broken_wing_butterflys(exp_groups, current_price):
    """Identifies and creates Broken Wing Butterfly (BWB) strategies.
    A BWB consists of 1 long, 2 short, 1 long options of the same type (call or put).
    It has a 'broken wing' where one spread is wider than the other.
    We are specifically looking for 'omnidirectional' BWBs where the extra short leg's
    credit is more than the max loss of the narrower spread.
    """
    broken_wing_butterflys = []
    current_price = float(current_price)

    for exp_date, group in exp_groups:
        long_calls = group[(group["option_type"] == "C") & (group["position_type"].str.contains("Long"))].sort_values("strike_price")
        short_calls = group[(group["option_type"] == "C") & (group["position_type"].str.contains("Short"))].sort_values("strike_price")
        long_puts = group[(group["option_type"] == "P") & (group["position_type"].str.contains("Long"))].sort_values("strike_price")
        short_puts = group[(group["option_type"] == "P") & (group["position_type"].str.contains("Short"))].sort_values("strike_price")

        notional_multiplier = float(group["notional_multiplier"].iloc[0]) if not group.empty else 1.0
        standard_deviation = float(group["standard_deviation"].iloc[0]) if not group.empty else np.nan

        # --- Call Broken Wing Butterfly (Long A, Short 2 B, Long C) ---
        # A < B < C, and (C-B) > (B-A) for a credit BWB (wider upper wing)
        for i, long_a in long_calls.iterrows():
            for j, short_b in short_calls.iterrows():
                for k, long_c in long_calls.iterrows():
                    if (
                        long_a["expiration_date"] == short_b["expiration_date"] == long_c["expiration_date"] == exp_date
                        and long_a["option_type"] == short_b["option_type"] == long_c["option_type"] == "C"
                        and long_a["strike_price"] < short_b["strike_price"] < long_c["strike_price"]
                    ):
                        strike_A, strike_B, strike_C = long_a["strike_price"], short_b["strike_price"], long_c["strike_price"]
                        mid_A, mid_B, mid_C = long_a["mid_price"], short_b["mid_price"], long_c["mid_price"]

                        narrow_wing_width = strike_B - strike_A
                        wide_wing_width = strike_C - strike_B

                        # Condition for a credit Call BWB (wider upper wing)
                        if not (wide_wing_width > narrow_wing_width):
                            continue

                        net_credit = (2 * mid_B) - mid_A - mid_C
                        if net_credit <= 0:  # Ensure it's a net credit BWB
                            continue

                        # Calculate max loss of the narrower spread (Long A, Short B)
                        # This is the maximum loss if you only traded the long vertical spread (Long A, Short B)
                        max_loss_narrower_spread = narrow_wing_width - (mid_B - mid_A)

                        # Omnidirectional condition: extra short leg's credit > max loss of the narrower spread
                        # P_B > (B-A) - (P_B - P_A)  =>  2 * P_B > (B-A) + P_A
                        if not ((2 * mid_B) > (strike_B - strike_A) + mid_A):
                            continue

                        # Max profit occurs at strike B
                        max_profit_value = (net_credit + narrow_wing_width) * notional_multiplier

                        # Max loss occurs if price goes above C (wider wing)
                        # The total risk is limited to the width of the wider spread minus the net credit.
                        max_loss_value = max(0, (wide_wing_width - net_credit)) * notional_multiplier

                        # Breakeven points
                        # Lower breakeven is effectively eliminated by the omnidirectional condition (or below A)
                        lower_breakeven = np.nan
                        # Upper breakeven: strike_C - (wide_wing_width - net_credit)
                        upper_breakeven = strike_C - (wide_wing_width - net_credit)

                        pop = calculate_probability_of_profit(
                            current_price,
                            (lower_breakeven, upper_breakeven),
                            standard_deviation,
                            "Broken Wing Butterfly",
                        )
                        if pop < 0.60:  # Filter for minimum probability of profit
                            continue

                        cvar = calculate_cvar(
                            "Broken Wing Butterfly",
                            max_loss_value / notional_multiplier,  # Pass max_loss per contract to CVaR
                            notional_multiplier,
                            pop=pop,
                            current_price=current_price,
                            standard_deviation=standard_deviation,
                        )

                        broken_wing_butterflys.append(
                            {
                                "expiration_date": exp_date,
                                "option_type": "C",
                                "strategy": "Call Broken Wing Butterfly (Omnidirectional)",
                                "strikes": f"{strike_A},{strike_B},{strike_C}",
                                "mid_price": net_credit,
                                "max_profit": max_profit_value,
                                "max_loss": max_loss_value,
                                "delta": long_a["delta"] - (2 * short_b["delta"]) + long_c["delta"],
                                "theta": long_a["theta"] - (2 * short_b["theta"]) + long_c["theta"],
                                "beta_weighted_delta": long_a["beta_weighted_delta"]
                                - (2 * short_b["beta_weighted_delta"])
                                + long_c["beta_weighted_delta"],
                                "position_type": f"Long Call {strike_A}/2x Short Call {strike_B}/Long Call {strike_C}",
                                "lower_breakeven": lower_breakeven,
                                "upper_breakeven": upper_breakeven,
                                "probability_of_profit": pop,
                                "cvar": cvar,
                            },
                        )

        # --- Put Broken Wing Butterfly (Long A, Short 2 B, Long C) ---
        # A < B < C, and (B-A) > (C-B) for a credit BWB (wider lower wing)
        for i, long_a in long_puts.iterrows():
            for j, short_b in short_puts.iterrows():
                for k, long_c in long_puts.iterrows():
                    if (
                        long_a["expiration_date"] == short_b["expiration_date"] == long_c["expiration_date"] == exp_date
                        and long_a["option_type"] == short_b["option_type"] == long_c["option_type"] == "P"
                        and long_a["strike_price"] < short_b["strike_price"] < long_c["strike_price"]
                    ):
                        strike_A, strike_B, strike_C = long_a["strike_price"], short_b["strike_price"], long_c["strike_price"]
                        mid_A, mid_B, mid_C = long_a["mid_price"], short_b["mid_price"], long_c["mid_price"]

                        narrow_wing_width = strike_C - strike_B
                        wide_wing_width = strike_B - strike_A

                        # Condition for a credit Put BWB (wider lower wing)
                        if not (wide_wing_width > narrow_wing_width):
                            continue

                        net_credit = (2 * mid_B) - mid_A - mid_C
                        if net_credit <= 0:  # Ensure it's a net credit BWB
                            continue

                        # Calculate max loss of the narrower spread (Short B, Long C)
                        # This is the maximum loss if you only traded the short vertical spread (Short B, Long C)
                        max_loss_narrower_spread = narrow_wing_width - (mid_B - mid_C)

                        # Omnidirectional condition: extra short leg's credit > max loss of the narrower spread
                        # P_B > (C-B) - (P_B - P_C)  =>  2 * P_B > (C-B) + P_C
                        if not ((2 * mid_B) > (strike_C - strike_B) + mid_C):
                            continue

                        # Max profit occurs at strike B
                        max_profit_value = (net_credit + narrow_wing_width) * notional_multiplier

                        # Max loss occurs if price goes below A (wider wing)
                        # The total risk is limited to the width of the wider spread minus the net credit.
                        max_loss_value = max(0, (wide_wing_width - net_credit)) * notional_multiplier

                        # Breakeven points
                        # Lower breakeven: strike_A + (wide_wing_width - net_credit)
                        lower_breakeven = strike_A + (wide_wing_width - net_credit)
                        # Upper breakeven is effectively eliminated by the omnidirectional condition (or above C)
                        upper_breakeven = np.nan

                        pop = calculate_probability_of_profit(
                            current_price,
                            (lower_breakeven, upper_breakeven),
                            standard_deviation,
                            "Broken Wing Butterfly",
                        )
                        if pop < 0.60:  # Filter for minimum probability of profit
                            continue

                        cvar = calculate_cvar(
                            "Broken Wing Butterfly",
                            max_loss_value / notional_multiplier,  # Pass max_loss per contract to CVaR
                            notional_multiplier,
                            pop=pop,
                            current_price=current_price,
                            standard_deviation=standard_deviation,
                        )

                        broken_wing_butterflys.append(
                            {
                                "expiration_date": exp_date,
                                "option_type": "P",
                                "strategy": "Put Broken Wing Butterfly (Omnidirectional)",
                                "strikes": f"{strike_A},{strike_B},{strike_C}",
                                "mid_price": net_credit,
                                "max_profit": max_profit_value,
                                "max_loss": max_loss_value,
                                "delta": long_a["delta"] - (2 * short_b["delta"]) + long_c["delta"],
                                "theta": long_a["theta"] - (2 * short_b["theta"]) + long_c["theta"],
                                "beta_weighted_delta": long_a["beta_weighted_delta"]
                                - (2 * short_b["beta_weighted_delta"])
                                + long_c["beta_weighted_delta"],
                                "position_type": f"Long Put {strike_A}/2x Short Put {strike_B}/Long Put {strike_C}",
                                "lower_breakeven": lower_breakeven,
                                "upper_breakeven": upper_breakeven,
                                "probability_of_profit": pop,
                                "cvar": cvar,
                            },
                        )
    return broken_wing_butterflys


def create_calendar_spreads(df, current_price):
    """Identifies and creates Calendar Spreads (same strike, different expirations)."""
    calendar_spreads = []
    current_price = float(current_price)

    # Group options by strike price and option type
    grouped_by_strike_type = df.groupby(["strike_price", "option_type"])

    for (strike_price, opt_type), group in grouped_by_strike_type:
        # Sort by expiration date (nearest to farthest)
        group = group.sort_values("expiration_date")

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                front_opt = group.iloc[i]  # Shorter expiration
                back_opt = group.iloc[j]  # Longer expiration

                # Ensure different expirations
                if front_opt["expiration_date"] == back_opt["expiration_date"]:
                    continue

                notional_multiplier = float(front_opt["notional_multiplier"])
                standard_deviation_front = float(front_opt["standard_deviation"])
                standard_deviation_back = float(back_opt["standard_deviation"])

                # We'll use the standard deviation of the farther expiration for simplicity
                # as it's typically the more dominant factor in risk.
                strategy_std_dev = standard_deviation_back

                # Long Calendar Spread: Sell nearer-term, Buy farther-term
                # Net Debit is expected
                net_debit = back_opt["mid_price"] - front_opt["mid_price"]

                if net_debit > 0:  # Check for net debit
                    strategy_name = f"Long Calendar {opt_type} (Debit)"
                    max_loss = net_debit * notional_multiplier  # Max loss is initial debit
                    # Max profit is variable and depends on implied volatility changes at expiration
                    # and the price of the back-month option at the front-month's expiration.
                    # It's challenging to accurately estimate without a full option pricing model.
                    max_profit = np.inf  # Placeholder for theoretical unbounded profit or highly variable profit.

                    # Breakeven points are complex due to different expirations and IV dynamics
                    # and require a more advanced pricing model or simulation.
                    lower_breakeven = np.nan
                    upper_breakeven = np.nan

                    # POP is hard to define accurately for calendar spreads with this model, setting to NaN.
                    pop = np.nan

                    cvar = calculate_cvar(
                        "Long Calendar",
                        net_debit,
                        notional_multiplier,
                        pop=pop,
                        current_price=current_price,
                        standard_deviation=strategy_std_dev,
                    )

                    calendar_spreads.append(
                        {
                            "expiration_date": front_opt["expiration_date"],  # Use front expiration for display
                            "option_type": opt_type,
                            "strategy": strategy_name,
                            "strikes": f"{strike_price}",  # Only one strike
                            "mid_price": -net_debit,  # Store as negative to indicate debit
                            "max_profit": max_profit,
                            "max_loss": max_loss,
                            "delta": back_opt["delta"] + front_opt["delta"],  # Simplified combined delta
                            "theta": back_opt["theta"] + front_opt["theta"],  # Combined theta
                            "beta_weighted_delta": back_opt["beta_weighted_delta"] + front_opt["beta_weighted_delta"],
                            "position_type": f"Long {opt_type} {strike_price} ({back_opt['expiration_date'].strftime('%Y-%m-%d')})/Short {opt_type} {strike_price} ({front_opt['expiration_date'].strftime('%Y-%m-%d')})",
                            "lower_breakeven": lower_breakeven,
                            "upper_breakeven": upper_breakeven,
                            "probability_of_profit": pop,
                            "cvar": cvar,
                        },
                    )
                else:  # Short Calendar Spread: Buy nearer-term, Sell farther-term (Net Credit is expected)
                    strategy_name = f"Short Calendar {opt_type} (Credit)"
                    net_credit = abs(net_debit)  # net_debit was negative, so abs is the credit
                    max_profit = net_credit * notional_multiplier  # Max profit is initial credit

                    # Max loss is potentially unlimited if price moves significantly away from strike and IV changes.
                    # This strategy typically has defined max profit and unlimited max loss.
                    max_loss = np.inf  # Potentially unlimited

                    lower_breakeven = np.nan  # Complex
                    upper_breakeven = np.nan  # Complex
                    pop = np.nan  # POP is hard to define accurately

                    cvar = calculate_cvar(
                        "Short Calendar",
                        max_loss / notional_multiplier,  # Pass max loss per contract to CVaR as a placeholder
                        notional_multiplier,
                        pop=pop,
                        current_price=current_price,
                        standard_deviation=strategy_std_dev,
                    )

                    calendar_spreads.append(
                        {
                            "expiration_date": front_opt["expiration_date"],  # Use front expiration for display
                            "option_type": opt_type,
                            "strategy": strategy_name,
                            "strikes": f"{strike_price}",
                            "mid_price": net_credit,
                            "max_profit": max_profit,
                            "max_loss": max_loss,
                            "delta": front_opt["delta"] + back_opt["delta"],  # Combined delta
                            "theta": front_opt["theta"] + back_opt["theta"],  # Combined theta
                            "beta_weighted_delta": front_opt["beta_weighted_delta"] + back_opt["beta_weighted_delta"],
                            "position_type": f"Short {opt_type} {strike_price} ({back_opt['expiration_date'].strftime('%Y-%m-%d')})/Long {opt_type} {strike_price} ({front_opt['expiration_date'].strftime('%Y-%m-%d')})",
                            "lower_breakeven": lower_breakeven,
                            "upper_breakeven": upper_breakeven,
                            "probability_of_profit": pop,
                            "cvar": cvar,
                        },
                    )

    return calendar_spreads


def create_diagonal_spreads(df, current_price):
    """Identifies and creates Diagonal Spreads (different strikes, different expirations).
    Includes both Long Diagonal (debit) and Short Diagonal (credit) spreads.
    """
    diagonal_spreads = []
    current_price = float(current_price)

    # Filter for options with different expiration dates
    exp_dates = sorted(df["expiration_date"].unique())

    for i in range(len(exp_dates)):
        front_exp_date = exp_dates[i]
        front_month_options = df[df["expiration_date"] == front_exp_date]

        for j in range(i + 1, len(exp_dates)):
            back_exp_date = exp_dates[j]
            back_month_options = df[df["expiration_date"] == back_exp_date]

            notional_multiplier = float(df["notional_multiplier"].iloc[0])  # Assume consistent
            standard_deviation = float(df["standard_deviation"].iloc[0])  # Use a representative std dev of one of the options

            # --- Long Diagonal Calls (Buy Back-Month, Lower Strike Call; Sell Front-Month, Higher Strike Call) ---
            # Ideal for bullish or moderately bullish outlook, profiting from time decay and modest upward move
            # Max Loss = Initial Debit Paid (defined risk).
            # Max Profit: The maximum profit is variable and cannot be precisely calculated without a
            # full option pricing model for the longer-dated option's remaining extrinsic value at
            # the front-month's expiration. It can be estimated as max intrinsic value + long option's remaining
            # extrinsic value. Setting to np.inf to reflect its variable/unbounded nature.
            for _, back_opt in back_month_options[back_month_options["option_type"] == "C"].iterrows():
                for _, front_opt in front_month_options[front_month_options["option_type"] == "C"].iterrows():
                    # Condition: back_opt strike < front_opt strike AND back_opt expiration > front_opt expiration
                    if back_opt["strike_price"] < front_opt["strike_price"]:
                        net_debit = back_opt["mid_price"] - front_opt["mid_price"]

                        # Only consider net debit diagonals
                        if net_debit <= 0:
                            continue

                        strategy_name = "Long Diagonal Call"
                        max_loss = net_debit * notional_multiplier  # Max loss is the debit paid
                        max_profit = np.inf  # Max profit is estimated but not precisely fixed.

                        # Breakeven price (at expiration of the front month): long call strike + debit paid
                        lower_breakeven = back_opt["strike_price"] + net_debit
                        upper_breakeven = np.nan  # No simple upper breakeven for this type of spread.

                        # POP for Long Diagonal Call: Price > lower_breakeven at front expiration
                        pop = calculate_probability_of_profit(current_price, lower_breakeven, standard_deviation, "Long Call")

                        cvar = calculate_cvar(
                            "Long Diagonal",
                            net_debit,  # For defined-risk debit strategies, CVaR is typically the initial debit.
                            notional_multiplier,
                            pop=pop,
                            current_price=current_price,
                            standard_deviation=standard_deviation,
                        )

                        diagonal_spreads.append(
                            {
                                "expiration_date": front_exp_date,  # Use front expiration for display
                                "option_type": "C",
                                "strategy": strategy_name,
                                "strikes": f"Long Call {back_opt['strike_price']} ({back_exp_date.strftime('%Y-%m-%d')}), Short Call {front_opt['strike_price']} ({front_exp_date.strftime('%Y-%m-%d')})",
                                "mid_price": -net_debit,  # Store as negative for debit
                                "max_profit": max_profit,
                                "max_loss": max_loss,
                                "delta": back_opt["delta"] - front_opt["delta"],
                                "theta": back_opt["theta"] - front_opt["theta"],
                                "beta_weighted_delta": back_opt["beta_weighted_delta"] - front_opt["beta_weighted_delta"],
                                "position_type": f"Long Call {back_opt['strike_price']} ({back_exp_date.strftime('%Y-%m-%d')})/Short Call {front_opt['strike_price']} ({front_exp_date.strftime('%Y-%m-%d')})",
                                "lower_breakeven": lower_breakeven,
                                "upper_breakeven": upper_breakeven,
                                "probability_of_profit": pop,
                                "cvar": cvar,
                            },
                        )

            # --- Long Diagonal Puts (Buy Back-Month, Higher Strike Put; Sell Front-Month, Lower Strike Put) ---
            # Ideal for bearish or moderately bearish outlook
            # Max Loss = Initial Debit Paid (defined risk).
            # Max Profit: The maximum profit is variable and cannot be precisely calculated without a
            # full option pricing model for the longer-dated option's remaining extrinsic value at
            # the front-month's expiration. Setting to np.inf to reflect its variable/unbounded nature.
            for _, back_opt in back_month_options[back_month_options["option_type"] == "P"].iterrows():
                for _, front_opt in front_month_options[front_month_options["option_type"] == "P"].iterrows():
                    # Condition: back_opt strike > front_opt strike AND back_opt expiration > front_opt expiration
                    if back_opt["strike_price"] > front_opt["strike_price"]:
                        net_debit = back_opt["mid_price"] - front_opt["mid_price"]

                        # Only consider net debit diagonals
                        if net_debit <= 0:
                            continue

                        strategy_name = "Long Diagonal Put"
                        max_loss = net_debit * notional_multiplier  # Max loss is the debit paid
                        max_profit = np.inf  # Max profit is estimated but not precisely fixed.

                        # Breakeven price (at expiration of the front month): long put strike - debit paid
                        lower_breakeven = np.nan  # No simple lower breakeven for this type of spread.
                        upper_breakeven = back_opt["strike_price"] - net_debit
                        pop = calculate_probability_of_profit(current_price, upper_breakeven, standard_deviation, "Long Put")

                        cvar = calculate_cvar(
                            "Long Diagonal",
                            net_debit,  # For defined-risk debit strategies, CVaR is typically the initial debit.
                            notional_multiplier,
                            pop=pop,
                            current_price=current_price,
                            standard_deviation=standard_deviation,
                        )

                        diagonal_spreads.append(
                            {
                                "expiration_date": front_exp_date,
                                "option_type": "P",
                                "strategy": strategy_name,
                                "strikes": f"Long Put {back_opt['strike_price']} ({back_exp_date.strftime('%Y-%m-%d')}), Short Put {front_opt['strike_price']} ({front_exp_date.strftime('%Y-%m-%d')})",
                                "mid_price": -net_debit,
                                "max_profit": max_profit,
                                "max_loss": max_loss,
                                "delta": back_opt["delta"] - front_opt["delta"],
                                "theta": back_opt["theta"] - front_opt["theta"],
                                "beta_weighted_delta": back_opt["beta_weighted_delta"] - front_opt["beta_weighted_delta"],
                                "position_type": f"Long Put {back_opt['strike_price']} ({back_exp_date.strftime('%Y-%m-%d')})/Short Put {front_opt['strike_price']} ({front_exp_date.strftime('%Y-%m-%d')})",
                                "lower_breakeven": lower_breakeven,
                                "upper_breakeven": upper_breakeven,
                                "probability_of_profit": pop,
                                "cvar": cvar,
                            },
                        )

            # --- Short Diagonal Calls (Sell Front-Month, Lower Strike Call; Buy Back-Month, Higher Strike Call) ---
            # Max Profit = Net Credit Received.
            # Max Loss = Unlimited (if held unmanaged beyond front-month expiration and price moves unfavorably).
            for _, front_opt in front_month_options[front_month_options["option_type"] == "C"].iterrows():
                for _, back_opt in back_month_options[back_month_options["option_type"] == "C"].iterrows():
                    # Condition: front_opt strike < back_opt strike AND back_opt expiration > front_opt expiration
                    if front_opt["strike_price"] < back_opt["strike_price"]:
                        net_credit = front_opt["mid_price"] - back_opt["mid_price"]

                        # Only consider net credit diagonals
                        if net_credit <= 0:
                            continue

                        strategy_name = "Short Diagonal Call"
                        max_profit = net_credit * notional_multiplier
                        max_loss = np.inf  # Potentially unlimited risk

                        # Breakeven points for diagonal spreads are complex and require advanced modeling.
                        # Setting to NaN.
                        lower_breakeven = np.nan
                        upper_breakeven = np.nan
                        pop = np.nan  # POP is also complex for diagonals, setting to NaN.

                        cvar = calculate_cvar(
                            "Short Diagonal",
                            max_loss / notional_multiplier,  # Pass max loss per contract to CVaR
                            notional_multiplier,
                            pop=pop,
                            current_price=current_price,
                            standard_deviation=standard_deviation,
                        )

                        diagonal_spreads.append(
                            {
                                "expiration_date": front_exp_date,
                                "option_type": "C",
                                "strategy": strategy_name,
                                "strikes": f"Short Call {front_opt['strike_price']} ({front_exp_date.strftime('%Y-%m-%d')}), Long Call {back_opt['strike_price']} ({back_exp_date.strftime('%Y-%m-%d')})",
                                "mid_price": net_credit,
                                "max_profit": max_profit,
                                "max_loss": max_loss,
                                "delta": front_opt["delta"] - back_opt["delta"],
                                "theta": front_opt["theta"] - back_opt["theta"],
                                "beta_weighted_delta": front_opt["beta_weighted_delta"] - back_opt["beta_weighted_delta"],
                                "position_type": f"Short Call {front_opt['strike_price']} ({front_exp_date.strftime('%Y-%m-%d')})/Long Call {back_opt['strike_price']} ({back_exp_date.strftime('%Y-%m-%d')})",
                                "lower_breakeven": lower_breakeven,
                                "upper_breakeven": upper_breakeven,
                                "probability_of_profit": pop,
                                "cvar": cvar,
                            },
                        )

            # --- Short Diagonal Puts (Sell Front-Month, Higher Strike Put; Buy Back-Month, Lower Strike Put) ---
            # Max Profit = Net Credit Received.
            # Max Loss = Unlimited (if held unmanaged beyond front-month expiration and price moves unfavorably).
            for _, front_opt in front_month_options[front_month_options["option_type"] == "P"].iterrows():
                for _, back_opt in back_month_options[back_month_options["option_type"] == "P"].iterrows():
                    # Condition: front_opt strike > back_opt strike AND back_opt expiration > front_opt expiration
                    if front_opt["strike_price"] > back_opt["strike_price"]:
                        net_credit = front_opt["mid_price"] - back_opt["mid_price"]

                        # Only consider net credit diagonals
                        if net_credit <= 0:
                            continue

                        strategy_name = "Short Diagonal Put"
                        max_profit = net_credit * notional_multiplier
                        max_loss = np.inf  # Potentially unlimited risk

                        # Breakeven points for diagonal spreads are complex and require advanced modeling.
                        # Setting to NaN.
                        lower_breakeven = np.nan
                        upper_breakeven = np.nan
                        pop = np.nan  # POP is also complex for diagonals, setting to NaN.

                        cvar = calculate_cvar(
                            "Short Diagonal",
                            max_loss / notional_multiplier,  # Pass max loss per contract to CVaR
                            notional_multiplier,
                            pop=pop,
                            current_price=current_price,
                            standard_deviation=standard_deviation,
                        )

                        diagonal_spreads.append(
                            {
                                "expiration_date": front_exp_date,
                                "option_type": "P",
                                "strategy": strategy_name,
                                "strikes": f"Short Put {front_opt['strike_price']} ({front_exp_date.strftime('%Y-%m-%d')}), Long Put {back_opt['strike_price']} ({back_exp_date.strftime('%Y-%m-%d')})",
                                "mid_price": net_credit,
                                "max_profit": max_profit,
                                "max_loss": max_loss,
                                "delta": front_opt["delta"] - back_opt["delta"],
                                "theta": front_opt["theta"] - back_opt["theta"],
                                "beta_weighted_delta": front_opt["beta_weighted_delta"] - back_opt["beta_weighted_delta"],
                                "position_type": f"Short Put {front_opt['strike_price']} ({front_exp_date.strftime('%Y-%m-%d')})/Long Put {back_opt['strike_price']} ({back_exp_date.strftime('%Y-%m-%d')})",
                                "lower_breakeven": lower_breakeven,
                                "upper_breakeven": upper_breakeven,
                                "probability_of_profit": pop,
                                "cvar": cvar,
                            },
                        )
    return diagonal_spreads


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
    import asyncio

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
        print(f"\nTotal calendar spreads created: {len(calendar_spreads_df)}")
    else:
        print("No calendar spreads created.")

    print("\n--- Diagonal Spreads ---")
    if diagonal_spreads_list:
        diagonal_spreads_df = pd.DataFrame(diagonal_spreads_list)
        print(diagonal_spreads_df.head())
        print(f"\nTotal diagonal spreads created: {len(diagonal_spreads_df)}")
    else:
        print("No diagonal spreads created.")
