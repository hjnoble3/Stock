import numpy as np
import pandas as pd
from scipy.stats import norm

# Configuration constant - assuming this is needed by calculate_cvar
CONFIDENCE_LEVEL = 0.95  # For CVaR calculation (e.g., 95% means we care about the worst 5% of outcomes)


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


def create_strangles(exp_groups, current_price):
    """Identifies and creates short strangle strategies."""
    strangles = []
    current_price = float(current_price)

    for exp_date, group in exp_groups:
        # Filter for short calls and short puts within the current expiration group
        # Assuming 'position_type' column is available after calling create_positions
        short_calls = group[(group["option_type"] == "C") & (group["position_type"].str.contains("Short"))].sort_values("strike_price")
        short_puts = group[(group["option_type"] == "P") & (group["position_type"].str.contains("Short"))].sort_values("strike_price")

        if short_calls.empty or short_puts.empty:
            continue

        notional_multiplier = float(group["notional_multiplier"].iloc[0]) if not group.empty else 1.0
        # Assuming 'standard_deviation' is available in the DataFrame
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