#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple


DEFAULT_MODULI: Tuple[int, ...] = (8, 9, 16, 25, 27, 32, 49)


@dataclass(frozen=True)
class SearchHit:
    a: int
    x: int
    b: int
    y: int
    c: int
    z: int
    gcd_abc: int

    @property
    def is_counterexample(self) -> bool:
        return self.gcd_abc == 1


def gcd3(a: int, b: int, c: int) -> int:
    return math.gcd(math.gcd(a, b), c)


def integer_nth_root(value: int, n: int) -> int:
    if value < 0:
        raise ValueError("value must be non-negative")
    if value < 2:
        return value
    low, high = 1, 1
    while pow(high, n) <= value:
        high *= 2
    while low + 1 < high:
        mid = (low + high) // 2
        if pow(mid, n) <= value:
            low = mid
        else:
            high = mid
    return low


def is_exact_nth_power(value: int, n: int) -> Tuple[bool, int]:
    root = integer_nth_root(value, n)
    return pow(root, n) == value, root


def power_residues(exponent: int, modulus: int) -> Set[int]:
    return {pow(a, exponent, modulus) for a in range(modulus)}


def congruence_summary(x: int, y: int, z: int, moduli: Sequence[int]) -> List[dict]:
    results: List[dict] = []
    for modulus in moduli:
        x_res = power_residues(x, modulus)
        y_res = power_residues(y, modulus)
        z_res = power_residues(z, modulus)
        pair_count = len(x_res) * len(y_res)
        survivors = 0
        for rx in x_res:
            for ry in y_res:
                if (rx + ry) % modulus in z_res:
                    survivors += 1
        ratio = survivors / pair_count if pair_count else 0.0
        results.append(
            {
                "modulus": modulus,
                "x_residue_count": len(x_res),
                "y_residue_count": len(y_res),
                "z_residue_count": len(z_res),
                "survivors": survivors,
                "pair_count": pair_count,
                "survival_ratio": ratio,
                "impossible": survivors == 0,
            }
        )
    return results


def analyze_exponent_triple(x: int, y: int, z: int, moduli: Sequence[int]) -> dict:
    summaries = congruence_summary(x, y, z, moduli)
    impossible = [item for item in summaries if item["impossible"]]
    average_ratio = sum(item["survival_ratio"] for item in summaries) / len(summaries)
    parity_note = parity_observation(x, y, z)
    return {
        "triple": [x, y, z],
        "average_survival_ratio": average_ratio,
        "parity_note": parity_note,
        "impossible_moduli": [item["modulus"] for item in impossible],
        "moduli": summaries,
    }


def parity_observation(x: int, y: int, z: int) -> str:
    if x % 2 == 0 and y % 2 == 0 and z % 2 == 0:
        return "三个指数都为偶数，可优先尝试降幂或公因子结构分析。"
    if x % 2 == 1 and y % 2 == 1 and z % 2 == 1:
        return "三个指数都为奇数时，奇偶性约束通常较弱，适合优先做计算搜索。"
    if z % 2 == 0:
        return "右侧为偶次幂，平方剩余/高次剩余过滤通常更有效。"
    return "混合奇偶指数，建议结合模 8、模 16 与奇素数幂模约束。"


def exponent_triples(exp_min: int, exp_max: int, symmetric: bool) -> Iterable[Tuple[int, int, int]]:
    space = range(exp_min, exp_max + 1)
    for x, y, z in itertools.product(space, repeat=3):
        if symmetric and x > y:
            continue
        yield x, y, z


def recommend_triples(
    exp_min: int,
    exp_max: int,
    moduli: Sequence[int],
    mode: str,
    top: int,
    symmetric: bool,
) -> List[dict]:
    ranked: List[dict] = []
    for x, y, z in exponent_triples(exp_min, exp_max, symmetric):
        analysis = analyze_exponent_triple(x, y, z, moduli)
        impossible_count = len(analysis["impossible_moduli"])
        avg_ratio = analysis["average_survival_ratio"]
        
        # Heuristic penalty for "safe zones" already explored by humans
        safe_zone_penalty = 0.0
        if x < 10 and y < 10 and z < 10:
            safe_zone_penalty = 2.0  # Heavily penalize small exponents that are well-studied
            
        if mode == "counterexample":
            score = avg_ratio - impossible_count - safe_zone_penalty
        else:
            score = impossible_count + (1.0 - avg_ratio) - safe_zone_penalty
            
        ranked.append(
            {
                "triple": [x, y, z],
                "score": score,
                "average_survival_ratio": avg_ratio,
                "impossible_moduli": analysis["impossible_moduli"],
                "parity_note": analysis["parity_note"] + (" (注: 小指数区易为已知安全区)" if safe_zone_penalty > 0 else ""),
            }
        )
    reverse = mode == "counterexample"
    ranked.sort(
        key=lambda item: (
            item["score"],
            item["average_survival_ratio"],
            -len(item["impossible_moduli"]),
        ),
        reverse=reverse,
    )
    return ranked[:top]


def build_power_table(max_base: int, exponents: Sequence[int]) -> Dict[int, List[int]]:
    table: Dict[int, List[int]] = {}
    for exponent in exponents:
        table[exponent] = [0] + [pow(base, exponent) for base in range(1, max_base + 1)]
    return table


def search_hits(
    max_base: int,
    exp_min: int,
    exp_max: int,
    moduli: Sequence[int],
    limit: int,
    symmetric: bool,
    counterexamples_only: bool,
) -> dict:
    exponents = list(range(exp_min, exp_max + 1))
    power_table = build_power_table(max_base, exponents)
    quick_filters = {
        triple: analyze_exponent_triple(*triple, moduli)
        for triple in exponent_triples(exp_min, exp_max, symmetric)
    }
    hits: List[SearchHit] = []
    checked_pairs = 0
    skipped_by_moduli = 0
    skipped_by_gcd = 0

    for x, y, z in exponent_triples(exp_min, exp_max, symmetric):
        analysis = quick_filters[(x, y, z)]
        if analysis["impossible_moduli"]:
            skipped_by_moduli += 1
            continue
        ax = power_table[x]
        by = power_table[y]
        for a in range(1, max_base + 1):
            a_pow = ax[a]
            b_start = a if symmetric and x == y else 1
            for b in range(b_start, max_base + 1):
                pair_coprime = math.gcd(a, b) == 1
                if counterexamples_only and not pair_coprime:
                    skipped_by_gcd += 1
                    continue
                checked_pairs += 1
                total = a_pow + by[b]
                ok, c = is_exact_nth_power(total, z)
                if not ok or c < 1 or c > max_base:
                    continue
                g = gcd3(a, b, c)
                hit = SearchHit(a=a, x=x, b=b, y=y, c=c, z=z, gcd_abc=g)
                if counterexamples_only and not hit.is_counterexample:
                    continue
                hits.append(hit)
                if len(hits) >= limit:
                    return summarize_hits(
                        hits,
                        checked_pairs,
                        skipped_by_moduli,
                        skipped_by_gcd,
                        max_base,
                        exp_min,
                        exp_max,
                        symmetric,
                        counterexamples_only,
                    )
    return summarize_hits(
        hits,
        checked_pairs,
        skipped_by_moduli,
        skipped_by_gcd,
        max_base,
        exp_min,
        exp_max,
        symmetric,
        counterexamples_only,
    )


def summarize_hits(
    hits: Sequence[SearchHit],
    checked_pairs: int,
    skipped_by_moduli: int,
    skipped_by_gcd: int,
    max_base: int,
    exp_min: int,
    exp_max: int,
    symmetric: bool,
    counterexamples_only: bool,
) -> dict:
    serialized_hits = [
        {
            "a": hit.a,
            "x": hit.x,
            "b": hit.b,
            "y": hit.y,
            "c": hit.c,
            "z": hit.z,
            "gcd_abc": hit.gcd_abc,
            "is_counterexample": hit.is_counterexample,
        }
        for hit in hits
    ]
    return {
        "search_config": {
            "max_base": max_base,
            "exp_min": exp_min,
            "exp_max": exp_max,
            "symmetric": symmetric,
            "counterexamples_only": counterexamples_only,
        },
        "stats": {
            "checked_coprime_pairs": checked_pairs,
            "skipped_exponent_triples_by_moduli": skipped_by_moduli,
            "skipped_pairs_by_gcd": skipped_by_gcd,
            "hits": len(serialized_hits),
        },
        "hits": serialized_hits,
    }


def print_json(data: dict) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def parse_moduli(raw: str | None) -> Tuple[int, ...]:
    if not raw:
        return DEFAULT_MODULI
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def cmd_search(args: argparse.Namespace) -> None:
    result = search_hits(
        max_base=args.max_base,
        exp_min=args.exp_min,
        exp_max=args.exp_max,
        moduli=parse_moduli(args.moduli),
        limit=args.limit,
        symmetric=not args.no_symmetric,
        counterexamples_only=args.counterexamples_only,
    )
    print_json(result)


def cmd_analyze(args: argparse.Namespace) -> None:
    result = analyze_exponent_triple(
        x=args.x,
        y=args.y,
        z=args.z,
        moduli=parse_moduli(args.moduli),
    )
    print_json(result)


def cmd_recommend(args: argparse.Namespace) -> None:
    result = {
        "mode": args.mode,
        "recommendations": recommend_triples(
            exp_min=args.exp_min,
            exp_max=args.exp_max,
            moduli=parse_moduli(args.moduli),
            mode=args.mode,
            top=args.top,
            symmetric=not args.no_symmetric,
        ),
    }
    print_json(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="比尔猜想研究辅助工具：搜索、模约束分析、指数三元组推荐。"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    search = subparsers.add_parser("search", help="在给定范围内搜索候选或反例。")
    search.add_argument("--max-base", type=int, default=200, help="底数上界。")
    search.add_argument("--exp-min", type=int, default=3, help="指数下界。")
    search.add_argument("--exp-max", type=int, default=7, help="指数上界。")
    search.add_argument("--limit", type=int, default=20, help="最多输出多少个命中。")
    search.add_argument(
        "--counterexamples-only",
        action="store_true",
        help="只输出 gcd(A,B,C)=1 的潜在反例。",
    )
    search.add_argument(
        "--no-symmetric",
        action="store_true",
        help="不使用 A^x + B^y 的对称去重。",
    )
    search.add_argument(
        "--moduli",
        type=str,
        default=None,
        help="逗号分隔的模数列表，例如 8,9,16,25。",
    )
    search.set_defaults(func=cmd_search)

    analyze = subparsers.add_parser("analyze", help="分析某个指数三元组的模约束强度。")
    analyze.add_argument("--x", type=int, required=True, help="左侧第一个指数。")
    analyze.add_argument("--y", type=int, required=True, help="左侧第二个指数。")
    analyze.add_argument("--z", type=int, required=True, help="右侧指数。")
    analyze.add_argument(
        "--moduli",
        type=str,
        default=None,
        help="逗号分隔的模数列表，例如 8,9,16,25。",
    )
    analyze.set_defaults(func=cmd_analyze)

    recommend = subparsers.add_parser(
        "recommend",
        help="根据模约束强弱推荐优先研究的指数三元组。",
    )
    recommend.add_argument("--exp-min", type=int, default=3, help="指数下界。")
    recommend.add_argument("--exp-max", type=int, default=12, help="指数上界。")
    recommend.add_argument(
        "--mode",
        choices=("counterexample", "proof"),
        default="counterexample",
        help="counterexample 偏向找更难筛掉的组合；proof 偏向找更容易建立障碍的组合。",
    )
    recommend.add_argument("--top", type=int, default=15, help="返回前多少个组合。")
    recommend.add_argument(
        "--no-symmetric",
        action="store_true",
        help="不使用 x,y 对称去重。",
    )
    recommend.add_argument(
        "--moduli",
        type=str,
        default=None,
        help="逗号分隔的模数列表，例如 8,9,16,25。",
    )
    recommend.set_defaults(func=cmd_recommend)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
