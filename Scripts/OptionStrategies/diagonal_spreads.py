import numpy as np
import pandas as pd
from scipy.stats import norm

# Helper functions used by create_diagonal_spreads
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
        percentile_for_cvar_tail = 1 - 0.95 # Assuming CONFIDENCE_LEVEL = 0.95
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
        percentile_for_cvar_tail = 0.95 # Assuming CONFIDENCE_LEVEL = 0.95
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
        expected_price_lower_tail = expected_tail_value(current_price, standard_deviation, 1 - 0.95, "lower") # Assuming CONFIDENCE_LEVEL = 0.95
        # Calculate expected price in the upper tail (relevant for the short call leg)
        expected_price_upper_tail = expected_tail_value(current_price, standard_deviation, 0.95, "upper") # Assuming CONFIDENCE_LEVEL = 0.95

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