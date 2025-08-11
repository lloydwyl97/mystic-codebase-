import random


def simulate_market_crash(df):
    df["close"] = df["close"] * (1 - random.uniform(0.1, 0.5))
    return df


def simulate_rug_pull(df):
    crash_index = random.randint(0, len(df) - 1)
    df.loc[crash_index:, "close"] = df.loc[crash_index:, "close"] * 0.1
    return df


def run_simulation(df, mode="crash"):
    if mode == "crash":
        return simulate_market_crash(df)
    elif mode == "rug":
        return simulate_rug_pull(df)
    else:
        return df
