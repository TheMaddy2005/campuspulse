"""Download the UCI Student Performance dataset into data/raw/.

The committed CSVs are enough to run the project, but this script lets you
re-fetch the raw data from the source.
"""
import io
import zipfile
from pathlib import Path

import urllib.request

URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL, timeout=60) as resp:
        outer = zipfile.ZipFile(io.BytesIO(resp.read()))
    # The outer zip contains student.zip, which holds the CSVs.
    with outer.open("student.zip") as inner_file:
        inner = zipfile.ZipFile(io.BytesIO(inner_file.read()))
    for name in ["student-mat.csv", "student-por.csv", "student.txt"]:
        target = RAW_DIR / name
        target.write_bytes(inner.read(name))
        print(f"  saved {target}")


if __name__ == "__main__":
    main()
