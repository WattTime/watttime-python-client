from watttime.api import WattTimeMyAccess
from watttime.evaluation.report import generate_report
from pathlib import Path
from typing import List, Tuple, Union, Literal
from collections import defaultdict

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


def bulk_generate_reports(
    model_date: str,
    signal_type: str,
    start: str,
    end: str,
    output_dir: Path,
    steps: List[Literal["signal", "fuel_mix", "forecast"]],
):
    regions = get_regions_for_model(model_date, signal_type)

    failed = []
    for region in regions:
        print(f"working on {region}")
        try:
            generate_report(
                region_list=region,
                signal_type=signal_type,
                model_date_list=[
                    get_latest_model_date(region, [model_date]),
                    model_date,
                ],
                eval_start=start,
                eval_end=end,
                steps=steps,
                output_dir=Path(output_dir),
            )
        except Exception as e:
            print(e)
            failed.append(region)
            continue


MODEL = "2024-10-01"
SIGNAL_TYPE = "co2_moer"
START = "2024-01-01"
END = "2024-12-31"
OUTPUT_DIR = Path("/home/skoeb/Desktop/watttime-python-client/analysis/2024-10-01")
if __name__ == "__main__":
    bulk_generate_reports(
        MODEL, SIGNAL_TYPE, START, END, OUTPUT_DIR, ["signal", "fuel_mix"]
    )
