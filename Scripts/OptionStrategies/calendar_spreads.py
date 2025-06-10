import numpy as np
import pandas as pd
from scipy.stats import norm


def calculate_probability_of_profit(
    current_price, breakeven, standard_deviation, option_type
):
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
            return norm.cdf(
                (upper_be - current_price) / standard_deviation
            ) - norm.cdf((lower_be - current_price) / standard_deviation)
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
    CONFIDENCE_LEVEL = 0.95  # Assuming this constant is needed here

    # Short Put: CVaR is the expected loss in the lower tail of the distribution.
    if strategy_type == "Short Put":
        if strike_leg1 is None:
            return np.nan
        # Determine the percentile for VaR for the lower tail (e.g., 0.05 for 95% CVaR)
        percentile_for_cvar_tail = 1 - CONFIDENCE_LEVEL
        expected_tail_price = expected_tail_value(
            current_price, standard_deviation, percentile_for_cvar_tail, "lower"
        )

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
        expected_tail_price = expected_tail_value(
            current_price, standard_deviation, percentile_for_cvar_tail, "upper"
        )

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
        expected_price_lower_tail = expected_tail_value(
            current_price, standard_deviation, 1 - CONFIDENCE_LEVEL, "lower"
        )
        # Calculate expected price in the upper tail (relevant for the short call leg)
        expected_price_upper_tail = expected_tail_value(
            current_price, standard_deviation, CONFIDENCE_LEVEL, "upper"
        )

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
                            "expiration_date": front_opt[
                                "expiration_date"
                            ],  # Use front expiration for display
                            "option_type": opt_type,
                            "strategy": strategy_name,
                            "strikes": f"{strike_price}",  # Only one strike
                            "mid_price": -net_debit,  # Store as negative to indicate debit
                            "max_profit": max_profit,
                            "max_loss": max_loss,
                            "delta": back_opt["delta"]
                            + front_opt["delta"],  # Simplified combined delta
                            "theta": back_opt["theta"] + front_opt["theta"],  # Combined theta
                            "beta_weighted_delta": back_opt["beta_weighted_delta"]
                            + front_opt["beta_weighted_delta"],
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
                        max_loss
                        / notional_multiplier,  # Pass max loss per contract to CVaR as a placeholder
                        notional_multiplier,
                        pop=pop,
                        current_price=current_price,
                        standard_deviation=strategy_std_dev,
                    )

                    calendar_spreads.append(
                        {
                            "expiration_date": front_opt[
                                "expiration_date"
                            ],  # Use front expiration for display
                            "option_type": opt_type,
                            "strategy": strategy_name,
                            "strikes": f"{strike_price}",
                            "mid_price": net_credit,
                            "max_profit": max_profit,
                            "max_loss": max_loss,
                            "delta": front_opt["delta"]
                            + back_opt["delta"],  # Combined delta
                            "theta": front_opt["theta"] + back_opt["theta"],  # Combined theta
                            "beta_weighted_delta": front_opt["beta_weighted_delta"]
                            + back_opt["beta_weighted_delta"],
                            "position_type": f"Short {opt_type} {strike_price} ({back_opt['expiration_date'].strftime('%Y-%m-%d')})/Long {opt_type} {strike_price} ({front_opt['expiration_date'].strftime('%Y-%m-%d')})",
                            "lower_breakeven": lower_breakeven,
                            "upper_breakeven": upper_breakeven,
                            "probability_of_profit": pop,
                            "cvar": cvar,
                        },
                    )

    return calendar_spreads