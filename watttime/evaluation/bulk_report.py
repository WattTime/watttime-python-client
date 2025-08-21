from watttime.api import WattTimeMyAccess
from watttime.evaluation.report import generate_report
from pathlib import Path
from typing import List, Tuple, Union, Literal
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

ma = WattTimeMyAccess()
MA_DF = ma.get_access_pandas()


def get_latest_model_date(abbrev, requested_model_dates: List[str]):
    if isinstance(abbrev, tuple):
        abbrev = abbrev[1]
    if isinstance(abbrev, list):
        abbrev = abbrev[0]
    models = list(
        set(
            MA_DF.loc[
                (MA_DF["signal_type"] == "co2_moer")
                & (MA_DF["region"] == abbrev)
                & (MA_DF["endpoint"] == "v3/historical")
            ]["model"]
        )
    )
    models = [i for i in models if i not in requested_model_dates]
    return max(models)


def get_regions_for_model(
    model: str, signal_type: str
) -> List[Union[str, Tuple[str, List[str]]]]:
    regions = MA_DF.loc[
        (MA_DF["signal_type"] == signal_type)
        & (MA_DF["model"] == model)
        & (MA_DF["endpoint"] == "v3/historical")
    ]["region"].unique()

    regions_dict = defaultdict(list)
    for region in regions:
        parent = region.split("_")[0]
        regions_dict[parent].append(region)

    regions_list = []
    for k, v in regions_dict.items():
        if len(v) > 1:
            regions_list.append((k, v))
        else:
            regions_list.append(v[0])

    return regions_list


def _process_region(region_list, signal_type, model_date, start, end, output_dir, steps):
    
    # TODO: make this DRY
    if isinstance(region_list, str):
        region_title = region_list
        region_list = [region_list]
    elif isinstance(region_list, tuple):
        region_title = region_list[0]
        region_list = region_list[1]
    else:
        region_title = "&".join(region_list)

    model_date_list = [
        get_latest_model_date(region_list, [model_date]),
        model_date,
    ]

    region_list = sorted(region_list)
    model_date_list = sorted(model_date_list)

    filename = f"{signal_type}_{region_title}_{'&'.join(model_date_list)}_model_stats"
    output_path = output_dir / f"{filename}.html"
    if output_path.exists():
        print(f"[SKIP] {output_path} already exists, skipping.")
        return (region_title, None)
    
    try:
        print(f"[PID] working on {region_title}")
        generate_report(
            region_list=region_list,
            signal_type=signal_type,
            model_date_list=model_date_list,
            eval_start=start,
            eval_end=end,
            steps=steps,
            output_dir=Path(output_dir),
            first_week_of_month_only=isinstance(region_title, str),
        )
        return (region_title, None)  # success
    except Exception as e:
        return (region_title, str(e))  # failure


def bulk_generate_reports(
    model_date: str,
    signal_type: str,
    start: str,
    end: str,
    output_dir: Path,
    steps: List[Literal["signal", "fuel_mix", "forecast"]],
    max_workers: int = 4,
):
    regions = get_regions_for_model(model_date, signal_type)
    failed = []

    if max_workers == 1:
        # Serial execution
        for region in regions:
            region, error = _process_region(
                region, signal_type, model_date, start, end, output_dir, steps
            )
            if error:
                print(f"[ERROR] {region}: {error}")
                failed.append(region)
    else:
        # Parallel execution
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_region = {
                executor.submit(
                    _process_region, region, signal_type, model_date, start, end, output_dir, steps
                ): region
                for region in regions
            }
            for future in as_completed(future_to_region):
                region, error = future.result()
                if error:
                    print(f"[ERROR] {region}: {error}")
                    failed.append(region)

    if failed:
        print("\nFailed regions:")
        for f in failed:
            print(f)


MODEL = "2024-10-01"
SIGNAL_TYPE = "co2_moer"
START = "2024-01-01T00:00Z"
END = "2024-12-31T00:00Z"
OUTPUT_DIR = Path("/app/watttime-python-client/analysis/2024-10-01")

if __name__ == "__main__":
    bulk_generate_reports(
        MODEL, SIGNAL_TYPE, START, END, OUTPUT_DIR, ["signal", "fuel_mix", "forecast"], max_workers=5
    )
