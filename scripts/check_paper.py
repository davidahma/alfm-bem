"""Check the final TeX pass and compare a distributed PDF with the current build."""
import argparse
import difflib
import re
import subprocess
import unicodedata
from pathlib import Path


def words(path):
    raw = subprocess.check_output(["pdftotext", "-raw", "-nopgbrk", str(path), "-"])
    text = unicodedata.normalize("NFKC", raw.decode("utf-8"))
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    text = re.sub(r"(?<=\w)-\n(?=[a-z])", "", text)
    return text.split()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", type=Path, help="also compare all extracted words")
    args = parser.parse_args()
    build = Path("build/paper")
    log = (build / "alfm-bem.log").read_text(errors="replace")
    warnings = re.findall(
        r"(?im)^.*(?:undefined references|citation .+ undefined|reference .+ undefined|"
        r"multiply defined|Overfull \\[hv]box|duplicate ignored|Rerun to get).*$", log)
    if warnings:
        raise SystemExit("\n".join(warnings))
    output = words(build / "alfm-bem.pdf")
    if len(output) < 1000 or "??" in output:
        raise SystemExit("Incomplete manuscript text or unresolved reference.")
    if args.compare:
        previous = words(args.compare)
        if previous != output:
            delta = difflib.unified_diff(previous, output, n=3, lineterm="")
            raise SystemExit("Distributed PDF text differs:\n" + "\n".join(list(delta)[:90]))
    print(f"Manuscript verified ({len(output)} words including appendices).")


if __name__ == "__main__":
    main()
