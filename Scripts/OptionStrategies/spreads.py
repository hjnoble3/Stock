import numpy as np
import pandas as pd
from scipy.stats import norm

# Helper functions used by create_spreads

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
    CONFIDENCE_LEVEL=0.95 # Assuming a default confidence level
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