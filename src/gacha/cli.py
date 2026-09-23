"""Command-line entry points: ``gacha evaluate`` and ``gacha experiment``."""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

from gacha.experiments import EXPERIMENTS, run_experiment
from gacha.kernel.chain import EnumeratedChain
from gacha.kernel.forward import hitting_time
from gacha.models.endfield import BannerState, SingleBannerModel
from gacha.models.plan import Plan
from gacha.risk import metrics as rk
from gacha.rules.endfield import BannerSpec, EndfieldCharacterRules


def _plan_from_toml(path: Path) -> tuple[Plan, Plan]:
    data = tomllib.loads(path.read_text())
    rules = EndfieldCharacterRules(**data.get("rules", {}))
    banners = [BannerSpec(rules, int(b["target_copies"]), int(b["cap"])) for b in data["banners"]]
    s = data.get("start", {})
    start = BannerState(
        int(s.get("t", 0)), int(s.get("n", 0)), int(s.get("c", 0)), int(s.get("u", 0))
    )
    current = Plan(banners, start=start, dossier0=bool(data.get("dossier0", False)))
    fresh = Plan(banners)
    return current, fresh


def _evaluate(args: argparse.Namespace) -> int:
    if args.plan:
        model, fresh = _plan_from_toml(Path(args.plan))
    else:
        spec = BannerSpec(EndfieldCharacterRules(), args.target, args.cap)
        model = SingleBannerModel(
            spec,
            start=BannerState(args.pity, args.banner_pulls, args.copies, args.up_obtained),
            dossier=bool(args.dossier),
        )
        fresh = SingleBannerModel(spec)
    ht = hitting_time(EnumeratedChain.from_model(model))
    budget = args.budget if args.budget is not None else ht.horizon
    lines = [
        f"P(success)                    {ht.p_success:.4f}",
        f"P(success within {budget} paid)    {rk.completion(ht, budget):.4f}",
        f"mean paid pulls               {rk.mean(ht):.2f}",
        f"sd                            {rk.sd(ht):.2f}",
    ]
    if ht.p_success > 0:
        for a in (0.5, 0.9, 0.95, 0.99):
            q = rk.quantile(ht, a, "success")
            lines.append(f"q{int(a * 100):<3d} (given success)         {q}")
        lines.append(f"CVaR90 (given success)        {rk.cvar(ht, 0.9, 'success'):.2f}")
    lines.append(f"expected excess over budget   {rk.expected_excess(ht, budget):.2f}")
    if args.realized is not None:
        fresh_ht = hitting_time(EnumeratedChain.from_model(fresh))
        surv = rk.survival(fresh_ht, args.realized)
        lines.append(f"P(T >= {args.realized}) from a fresh banner  {surv:.4f}")
    print("\n".join(lines))
    return 0


def _experiment(args: argparse.Namespace) -> int:
    names = EXPERIMENTS if args.name == "all" else [args.name]
    for name in names:
        if name not in EXPERIMENTS:
            known = ", ".join(EXPERIMENTS) or "(none)"
            print(f"unknown experiment {name!r}; known: {known}", file=sys.stderr)
            return 2
        results_md = Path(args.results_md) if args.results_md else None
        headline = run_experiment(name, Path(args.out), results_md=results_md)
        items = ", ".join(
            f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in headline.items()
        )
        print(f"{name}: {items}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gacha", description="Exact gacha waiting-time models")
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("evaluate", help="evaluate a personal banner state or a plan file")
    ev.add_argument("--pity", type=int, default=0, help="pity counter t (0-79)")
    ev.add_argument(
        "--banner-pulls", type=int, default=0, help="counted pulls already made on this banner"
    )
    ev.add_argument("--copies", type=int, default=0, help="UP copies already obtained here")
    ev.add_argument(
        "--up-obtained",
        type=int,
        choices=(0, 1),
        default=0,
        help="1 if the 120 guarantee is already void",
    )
    ev.add_argument(
        "--dossier",
        type=int,
        choices=(0, 1),
        default=0,
        help="1 if this banner received the 60-pull dossier",
    )
    ev.add_argument("--target", type=int, default=1, help="wanted UP copies")
    ev.add_argument("--cap", type=int, default=120, help="maximum paid pulls on this banner")
    ev.add_argument("--budget", type=int, default=None, help="paid pulls available")
    ev.add_argument(
        "--realized",
        type=int,
        default=None,
        help="paid pulls you actually needed; prints how unlucky that was",
    )
    ev.add_argument("--plan", type=str, default=None, help="TOML plan file (multi-banner)")
    ev.set_defaults(func=_evaluate)

    ex = sub.add_parser("experiment", help="run a registered experiment or 'all'")
    ex.add_argument("name")
    ex.add_argument("--out", default="results")
    ex.add_argument("--results-md", default="docs/results.md")
    ex.set_defaults(func=_experiment)

    args = parser.parse_args(argv)
    return args.func(args)
