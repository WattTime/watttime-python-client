import pandas as pd
import matplotlib.pyplot as plt
import evaluation.eval_framework as evu
import evaluation.battery as b

def make_test_moer_data():
    """Constant MOER at 1000.0 lbs/MWh except a single interval in the middle with 0.0"""
    start = pd.Timestamp("2023-12-13T09:45:00+00:00")
    end = pd.Timestamp("2023-12-13T11:40:00+00:00")
    moer_data = pd.DataFrame(columns=["point_time"], data=pd.date_range(start, end, freq="300s"))
    moer_data["value"] = 1000.0
    moer_data.loc[len(moer_data)//2, "value"] = 0.0 # middle interval has zero MOER
    return moer_data

def make_test_batteries():
    """Make three batteries which each have fast charging at a different point in their charging cycle"""
    batteries = []
    for i in range(1, 4):
        charging_curve = pd.DataFrame(
            columns=["SoC", "kW"],
            data=[
                [0.0, 10.0],
                [i / 5, 10.0],
                [i / 5 + 0.01, 50.0],
                [i / 5 + 0.20, 50.0],
                [i / 5 + 0.21, 10.0],
                [1.0, 10.0],
            ]
        )
        battery = b.Battery(
            initial_soc=0.0,
            charging_curve=charging_curve,
            capacity_kWh=10.0
        )
        batteries.append(battery)
    return batteries

def main():
    """
    Run the test with a combined plot and save the plot as test_output.png
    
    In this test, we construct three batteries, each of which have a "fast zone"
    of charging; at 20% to 40%, 40% to 60%, and 60% to 80% respectively.

    We also construct a test MOER forecast which has constant MOER except
    a single interval in the middle where the MOER drops to zero.

    We use the optimizer to get the schedule for each of these three batteries.
    The schedules will be different: the optimizer wants to charge in the fast zone
    during the zero MOER period. To do this, the battery with a later fast zone will
    have to charge more before the zero MOER period. This behavior does indeed occur
    and is shown in the output plot.
    """
    moer_data = make_test_moer_data()
    batteries = make_test_batteries()

    fig, axes = plt.subplots(3, len(batteries), figsize=(15, 10), gridspec_kw={'height_ratios': [1, 1, 2]})
    fig.suptitle("Test MOER Forecast, Battery Charging Curves. Output Optimizer Schedules")

    moer_data.set_index("point_time")["value"].plot(ax=axes[0, 1], title="MOER Forecast", grid=True)

    for i, battery in enumerate(batteries):
        # Plot the variable charging curve for each test battery
        battery.plot_charging_curve(ax=axes[1, i])
        
        # Calculate time_needed to plug into API
        time_needed = evu.get_time_needed(
            total_capacity_kWh=battery.capacity_kWh,
            usage_power_kW=battery.get_usage_power_kw_df(),
            initial_capacity_fraction=battery.initial_soc,
        )

        # Get schedule
        schedule = evu.get_schedule_and_cost_api(
            usage_power_kw=battery.get_usage_power_kw_df(),
            time_needed=time_needed,
            total_time_horizon=len(moer_data),
            moer_data=moer_data,
            optimization_method="auto",
        )

        # Plot schedule
        schedule["usage"].plot(ax=axes[2, i], title=f"Battery {i+1} Schedule", grid=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("test_output.png")

if __name__ == "__main__":
    main()