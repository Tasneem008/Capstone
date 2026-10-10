"""E9. Rerun the CV comparison and temporal validation on deduplicated rows."""

from experiments.e3_baselines_cv import run as run_cv
from experiments.e4_temporal_validation import run as run_temporal
from experiments.lib import deduplicate, load_frame


def main() -> None:
    frame = load_frame()
    kept, summary = deduplicate(frame)
    print("Dedup summary:", summary)
    run_cv(kept, tag="dedup")
    run_temporal(kept, tag="dedup")


if __name__ == "__main__":
    main()
