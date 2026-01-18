import pandas as pd
from pandas.api.types import is_numeric_dtype
from pathlib import Path


def main() -> None:
    path = Path("1-3/dataset/train.csv")
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")

    df = pd.read_csv(path)
    output_path = Path("1-3/test_summary.txt")

    lines = [f"Rows: {len(df)}, Cols: {len(df.columns)}"]
    for col in df.columns:
        lines.append(f"\n=== {col} ===")
        series = df[col]
        missing_count = int(series.isna().sum())
        lines.append(f"missing: {missing_count > 0}")
        if is_numeric_dtype(series):
            min_val = series.min(skipna=True)
            max_val = series.max(skipna=True)
            mean_val = series.mean(skipna=True)
            lines.append(f"min: {min_val}")
            lines.append(f"max: {max_val}")
            lines.append(f"mean: {mean_val}")
        else:
            unique_count = series.nunique(dropna=False)
            lines.append(f"unique_values: {unique_count}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote summary to: {output_path}")


if __name__ == "__main__":
    main()
