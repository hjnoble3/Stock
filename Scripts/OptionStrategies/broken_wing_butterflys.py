import numpy as np
import pandas as pd
from scipy.stats import norm

# Assume calculate_probability_of_profit and calculate_cvar are defined elsewhere
# and imported or available in the scope where this function is used.
# For independent operation in this file, including them here based on the prompt.


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
    CONFIDENCE_LEVEL=0.95, # Adding default here for independence
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
                                + (2 * short_b["beta_weighted_delta"])
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
                                + (2 * short_b["beta_weighted_delta"])
                                + long_c["beta_weighted_delta"],
                                "position_type": f"Long Put {strike_A}/2x Short Put {strike_B}/Long Put {strike_C}",
                                "lower_breakeven": lower_breakeven,
                                "upper_breakeven": upper_breakeven,
                                "probability_of_profit": pop,
                                "cvar": cvar,
                            },
                        )
    return broken_wing_butterflys