"""Render the primary results directly from the committed benchmark record."""
import argparse
import json
from pathlib import Path


def render(record):
    unit = record["summary"]["unit"]["methods"]
    runs = record["runs"]
    n = sum(run["available"]["future"]["n"] for run in runs)
    errors = sum(run["available"]["future"]["errors"] for run in runs)
    quota = sum(run["methods"]["frozen"]["future"]["unit"]["reviewed"]["n"] for run in runs)
    policies = [
        ("arrival", "Arrival order"),
        ("failure_only", "Nearest failure"),
        ("bem_sign", "BEM risk"),
        ("frozen", "Frozen confidence"),
        ("bem_sign_blend", "Confidence and BEM risk"),
        ("bem_joint_blend", "Joint BEM scores and confidence"),
        ("outcome_vote", "Confidence and outcome vote"),
        ("class_vote", "Confidence and class vote"),
    ]
    abstract = (
        f"Across {len(runs)} paired replays, review of {quota:,} out of {n:,} future "
        f"case presentations catches {unit['bem_sign']['total_removed']} of {errors} errors "
        f"using the BEM risk score, {unit['frozen']['total_removed']} using classifier confidence, "
        f"and {unit['class_vote']['total_removed']} using a calibrated combination of "
        "confidence and a local class vote. A calibrated policy combining BEM risk, "
        f"success, familiarity and confidence catches {unit['bem_joint_blend']['total_removed']}."
    )
    chart = [
        r"\paragraph{A common review budget.}",
        f"The frozen predictor made {errors} errors across {n:,} future case presentations. "
        f"Each policy referred {quota} cases. Figure~\\ref{{fig:review}} counts the errors "
        "caught by a perfect reviewer. Each seed reuses a subset of the same finite corpus. "
        "The totals are paired replay counts; the same digit can appear in more than one replay.",
        r"\begin{figure}[htbp]\centering",
        r"\begin{tabular}{@{}lrl@{}}\toprule",
        r"Review ranking & Errors caught & Share of underlying errors\\\midrule",
    ]
    for key, label in policies:
        caught = unit[key]["total_removed"]
        fraction = caught / errors
        chart.append(
            rf"{label} & {caught} & \textcolor{{teal!70!black}}{{\rule{{{fraction * 31:.2f}mm}}{{1.25ex}}}}"
            rf"\hspace{{0.45em}}{100 * fraction:.1f}\%\\"
        )
    chart.extend([
        r"\bottomrule\end{tabular}",
        r"\caption{Errors selected for review at identical capacity (five referrals per batch of fifty). "
        r"The votes combine local evidence with classifier confidence using held-out calibration. "
        r"Appendix~\ref{app:budgetref} gives the calibration protocol, uncertainty and cost-sensitive comparisons.}",
        r"\label{fig:review}\end{figure}",
    ])
    return {"abstract-result.tex": abstract, "primary-result.tex": "\n".join(chart)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    record = json.loads(Path("results/budget.json").read_text())
    for name, content in render(record).items():
        path = Path("paper", name)
        if args.check:
            if path.read_text() != content + "\n":
                raise SystemExit(f"{path} differs from the benchmark. Run python scripts/paper_results.py.")
        else:
            path.write_text(content + "\n")


if __name__ == "__main__":
    main()
