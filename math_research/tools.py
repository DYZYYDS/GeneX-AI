from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import dataclass
from typing import Any, Callable

from beal_tool import analyze_exponent_triple, power_residues, recommend_triples
from .proof_tree import proof_tree_manager

from .memory import MemoryStore
from .resource_manager import ResourceManager


ToolHandler = Callable[..., Any]


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    handler: ToolHandler


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, name: str, description: str, handler: ToolHandler) -> None:
        self._tools[name] = ToolDefinition(name=name, description=description, handler=handler)

    def execute(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"未知工具: {name}")
        handler = self._tools[name].handler
        cleaned = self._remap_aliases(kwargs)
        try:
            return handler(**cleaned)
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                import inspect
                sig = inspect.signature(handler)
                valid = {k for k in cleaned if k in sig.parameters}
                stripped = {k: v for k, v in cleaned.items() if k in valid}
                return handler(**stripped)
            raise

    def _remap_aliases(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        out = dict(kwargs)
        swaps = [
            ("statement", "proposition"), ("formula", "proposition"), ("expr", "expression"),
            ("variable", "symbol"), ("var", "symbol"), ("sym", "symbol"),
            ("lemma_id", "node_id"), ("id", "node_id"), ("name", "node_id"),
            ("root_statement", "description"), ("content", "description"), ("proof", "justification"),
            ("reason", "justification"), ("proof_summary", "justification"), ("new_status", "status"),
            ("equation_str", "equation"), ("eq", "equation"),
            ("matrix_data", "matrix_json"), ("matrix_a", "matrix_a_json"), ("matrix_b", "matrix_b_json"),
            ("rhs_data", "rhs_json"), ("coeffs", "coefficients_json"),
            ("left_expr", "expression_left"), ("right_expr", "expression_right"),
            ("subs", "substitutions_json"), ("substitutions", "substitutions_json"),
            ("expr1", "expression_left"), ("expr2", "expression_right"),
            ("polys", "expressions_json"), ("polynomials", "expressions_json"),
            ("symbols", "symbols_csv"), ("vars_csv", "symbols_csv"),
        ]
        for alias, canonical in swaps:
            if alias in out and canonical not in out:
                out[canonical] = out.pop(alias)
        # 最终兜底: proposition/statement 在任何函数里都应该映射到 expression (第一个字符串参数)
        if "proposition" in out and "expression" not in out:
            out["expression"] = out.pop("proposition")
        if "statement" in out and "expression" not in out:
            out["expression"] = out.pop("statement")
        return out

    def describe(self) -> list[dict[str, str]]:
        return [
            {"name": tool.name, "description": tool.description}
            for tool in sorted(self._tools.values(), key=lambda item: item.name)
        ]

    def names(self) -> list[str]:
        return sorted(self._tools)


def gcd_lcm(a: int, b: int) -> dict[str, int]:
    return {"gcd": math.gcd(a, b), "lcm": math.lcm(a, b)}


def prime_factorization(n: int) -> dict[str, Any]:
    if n == 0:
        return {"n": 0, "factors": {}}
    value = abs(n)
    factors: dict[int, int] = {}
    divisor = 2
    while divisor * divisor <= value:
        while value % divisor == 0:
            factors[divisor] = factors.get(divisor, 0) + 1
            value //= divisor
        divisor = 3 if divisor == 2 else divisor + 2
    if value > 1:
        factors[value] = factors.get(value, 0) + 1
    return {"n": n, "sign": -1 if n < 0 else 1, "factors": factors}


def p_adic_valuation(n: int, p: int) -> dict[str, int]:
    if p <= 1:
        raise ValueError("p 必须大于 1。")
    if n == 0:
        return {"n": n, "p": p, "valuation": -1}
    value = abs(n)
    count = 0
    while value % p == 0:
        value //= p
        count += 1
    return {"n": n, "p": p, "valuation": count, "remaining": value}


def modular_power(base: int, exponent: int, modulus: int) -> dict[str, int]:
    return {"value": pow(base, exponent, modulus)}


def residue_scan(exponent: int, modulus: int) -> dict[str, Any]:
    residues = sorted(power_residues(exponent, modulus))
    return {"exponent": exponent, "modulus": modulus, "residues": residues, "count": len(residues)}


def is_prime(n: int) -> dict[str, Any]:
    if n < 2:
        return {"n": n, "is_prime": False}
    if n % 2 == 0:
        return {"n": n, "is_prime": n == 2}
    divisor = 3
    while divisor * divisor <= n:
        if n % divisor == 0:
            return {"n": n, "is_prime": False, "witness": divisor}
        divisor += 2
    return {"n": n, "is_prime": True}


def divisors(n: int) -> dict[str, Any]:
    factors = prime_factorization(n)["factors"]
    result = [1]
    for prime, exponent in factors.items():
        current = list(result)
        for power in range(1, exponent + 1):
            result.extend(value * (prime**power) for value in current)
    values = sorted(set(result + ([-x for x in result] if n < 0 else [])))
    return {"n": n, "divisors": values, "count": len(values)}


def residue_pair_filter(x: int, y: int, z: int, modulus: int) -> dict[str, Any]:
    x_res = sorted(power_residues(x, modulus))
    y_res = sorted(power_residues(y, modulus))
    z_res = set(power_residues(z, modulus))
    survivors = [[rx, ry] for rx in x_res for ry in y_res if (rx + ry) % modulus in z_res]
    return {
        "x": x,
        "y": y,
        "z": z,
        "modulus": modulus,
        "survivors": survivors,
        "survivor_count": len(survivors),
        "pair_count": len(x_res) * len(y_res),
    }


def _has_sympy() -> bool:
    return importlib.util.find_spec("sympy") is not None


def _has_numpy() -> bool:
    return importlib.util.find_spec("numpy") is not None


def _has_scipy() -> bool:
    return importlib.util.find_spec("scipy") is not None


def sympy_available() -> dict[str, Any]:
    return {"available": _has_sympy()}


def numpy_available() -> dict[str, Any]:
    return {"available": _has_numpy()}


def scipy_available() -> dict[str, Any]:
    return {"available": _has_scipy()}


def _sympy_import() -> Any:
    if not _has_sympy():
        raise RuntimeError("当前环境未安装 sympy。")
    import sympy as sp

    return sp


def _numpy_import() -> Any:
    if not _has_numpy():
        raise RuntimeError("当前环境未安装 numpy。")
    import numpy as np

    return np


def _scipy_import() -> tuple[Any, Any]:
    if not _has_scipy():
        raise RuntimeError("当前环境未安装 scipy。")
    from scipy import linalg, optimize

    return linalg, optimize


def numeric_eval(expression: str, substitutions_json: str | None = None) -> dict[str, Any]:
    sp = _sympy_import()
    expr = sp.sympify(expression)
    subs = json.loads(substitutions_json) if substitutions_json else {}
    
    # 支持单组替换或多组替换
    if isinstance(subs, list):
        results = []
        for sub in subs:
            val = expr.subs({sp.Symbol(k): v for k, v in sub.items()})
            results.append(str(sp.N(val)))
        return {"input": expression, "results": results}
    else:
        val = expr.subs({sp.Symbol(k): v for k, v in subs.items()})
        return {"input": expression, "result": str(sp.N(val))}


def prime_sieve(limit: int) -> dict[str, Any]:
    if limit < 2:
        return {"limit": limit, "primes": [], "count": 0}
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[:2] = b"\x00\x00"
    for value in range(2, int(limit**0.5) + 1):
        if sieve[value]:
            start = value * value
            sieve[start : limit + 1 : value] = b"\x00" * (((limit - start) // value) + 1)
    primes = [idx for idx in range(limit + 1) if sieve[idx]]
    return {"limit": limit, "primes": primes, "count": len(primes)}


def sympy_simplify(expression: str) -> dict[str, Any]:
    sp = _sympy_import()
    expr = sp.sympify(expression)
    return {"input": expression, "result": str(sp.simplify(expr))}


def sympy_factor(expression: str) -> dict[str, Any]:
    sp = _sympy_import()
    expr = sp.sympify(expression)
    return {"input": expression, "result": str(sp.factor(expr))}


def sympy_expand(expression: str) -> dict[str, Any]:
    sp = _sympy_import()
    expr = sp.sympify(expression)
    return {"input": expression, "result": str(sp.expand(expr))}


def sympy_solve(equation: str, symbol: str) -> dict[str, Any]:
    sp = _sympy_import()
    sym = sp.Symbol(symbol)
    if "=" in equation:
        left, right = equation.split("=", 1)
        expr = sp.Eq(sp.sympify(left), sp.sympify(right))
    else:
        expr = sp.Eq(sp.sympify(equation), 0)
    solutions = sp.solve(expr, sym)
    return {"equation": equation, "symbol": symbol, "solutions": [str(item) for item in solutions]}


def sympy_diff(expression: str, symbol: str) -> dict[str, Any]:
    sp = _sympy_import()
    sym = sp.Symbol(symbol)
    expr = sp.sympify(expression)
    return {"input": expression, "symbol": symbol, "result": str(sp.diff(expr, sym))}


def sympy_integrate(expression: str, symbol: str) -> dict[str, Any]:
    sp = _sympy_import()
    sym = sp.Symbol(symbol)
    expr = sp.sympify(expression)
    return {"input": expression, "symbol": symbol, "result": str(sp.integrate(expr, sym))}


def sympy_series(expression: str, symbol: str, around: int = 0, order: int = 6) -> dict[str, Any]:
    sp = _sympy_import()
    sym = sp.Symbol(symbol)
    expr = sp.sympify(expression)
    return {
        "input": expression,
        "symbol": symbol,
        "around": around,
        "order": order,
        "result": str(sp.series(expr, sym, around, order)),
    }


def sympy_limit(expression: str, symbol: str, point: str) -> dict[str, Any]:
    sp = _sympy_import()
    sym = sp.Symbol(symbol)
    expr = sp.sympify(expression)
    return {"input": expression, "symbol": symbol, "point": point, "result": str(sp.limit(expr, sym, sp.sympify(point)))}


def sympy_nsolve(equation: str, symbol: str, guess: str) -> dict[str, Any]:
    sp = _sympy_import()
    sym = sp.Symbol(symbol)
    if "=" in equation:
        left, right = equation.split("=", 1)
        expr = sp.sympify(left) - sp.sympify(right)
    else:
        expr = sp.sympify(equation)
    result = sp.nsolve(expr, sym, sp.sympify(guess))
    return {"equation": equation, "symbol": symbol, "guess": guess, "result": str(result)}


def sympy_diophantine(equation: str) -> dict[str, Any]:
    sp = _sympy_import()
    expr = sp.sympify(equation)
    from sympy.solvers.diophantine import diophantine

    solutions = diophantine(expr)
    return {"equation": equation, "solutions": [str(item) for item in solutions]}


def sympy_groebner(expressions_json: str, symbols_csv: str, order: str = "lex") -> dict[str, Any]:
    sp = _sympy_import()
    expressions = [sp.sympify(item) for item in json.loads(expressions_json)]
    symbols = [sp.Symbol(item.strip()) for item in symbols_csv.split(",") if item.strip()]
    basis = sp.groebner(expressions, *symbols, order=order)
    return {"expressions": [str(item) for item in expressions], "symbols": symbols_csv, "basis": [str(item) for item in basis]}


def sympy_matrix_det(matrix_json: str) -> dict[str, Any]:
    sp = _sympy_import()
    matrix = sp.Matrix(json.loads(matrix_json))
    return {"matrix": matrix.tolist(), "determinant": str(matrix.det())}


def sympy_matrix_eigenvals(matrix_json: str) -> dict[str, Any]:
    sp = _sympy_import()
    matrix = sp.Matrix(json.loads(matrix_json))
    eigen = {str(key): int(value) for key, value in matrix.eigenvals().items()}
    return {"matrix": matrix.tolist(), "eigenvalues": eigen}


def sympy_is_identity(expression_left: str, expression_right: str) -> dict[str, Any]:
    sp = _sympy_import()
    left = sp.sympify(expression_left)
    right = sp.sympify(expression_right)
    delta = sp.simplify(left - right)
    return {"left": expression_left, "right": expression_right, "is_identity": delta == 0, "delta": str(delta)}


def z3_logic_prove(proposition: str) -> dict[str, Any]:
    """使用 Z3 SMT 求解器验证一个布尔/代数逻辑命题是否恒成立。
    如果返回 unsat，说明原命题的反面不可满足，即原命题是重言式（定理）。
    """
    try:
        import z3
    except ImportError:
        return {"error": "Z3 solver is not installed."}
        
    try:
        # 为了安全和简便，这里通过简单的 eval 解析命题。
        # 实际生产中应使用严格的解析器。这里只允许受限的符号声明。
        local_vars = {"z3": z3, "Implies": z3.Implies, "And": z3.And, "Or": z3.Or, "Not": z3.Not, "ForAll": z3.ForAll, "Exists": z3.Exists}
        # 自动声明单字母变量为 Z3 Bool 或 Int
        for char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
            local_vars[char] = z3.Bool(char)
        
        expr = eval(proposition, {"__builtins__": {}}, local_vars)
        solver = z3.Solver()
        # 证明 P 等价于证明 Not(P) 是 Unsat
        solver.add(z3.Not(expr))
        result = solver.check()
        
        if result == z3.unsat:
            return {"proposition": proposition, "result": "Proved (Tautology)"}
        elif result == z3.sat:
            model = solver.model()
            return {"proposition": proposition, "result": "Refuted (Counterexample exists)", "counterexample": str(model)}
        else:
            return {"proposition": proposition, "result": "Unknown"}
    except Exception as e:
        return {"error": f"解析或求解 Z3 命题失败: {str(e)}"}

def sympy_logic_prove(proposition: str) -> dict[str, Any]:
    """使用 sympy.logic 验证一个布尔逻辑命题是否为重言式(Tautology)"""
    sp = _sympy_import()
    from sympy.logic.boolalg import simplify_logic, is_tautology
    try:
        expr = sp.sympify(proposition)
        simplified = simplify_logic(expr)
        tautology = is_tautology(expr)
        return {
            "proposition": proposition,
            "simplified": str(simplified),
            "is_tautology": bool(tautology)
        }
    except Exception as e:
        return {"error": f"解析逻辑表达式失败: {str(e)}"}

def retrieve_known_mathematical_bounds(topic: str) -> dict[str, Any]:
    """模拟查询人类已知的数学穷举边界"""
    topic = topic.lower()
    if "beal" in topic or "比尔" in topic:
        return {
            "topic": "Beal Conjecture",
            "known_exhaustion_bounds": "人类已穷举验证底数 A, B, C <= 1000 且指数 x, y, z <= 100。在此范围内不存在满足 gcd(A,B,C)=1 的反例。",
            "implication": "不要使用计算工具搜索小于这个边界的数值空间，这是纯粹的算力浪费。"
        }
    if "collatz" in topic or "考拉兹" in topic or "冰雹" in topic:
        return {
            "topic": "Collatz Conjecture",
            "known_exhaustion_bounds": "已穷举验证所有 n < 2^68 (约 2.95×10^20) 都满足猜想。",
            "implication": "低于 2^68 的计算验证没有数学价值。"
        }
    if "riemann" in topic or "黎曼" in topic:
        return {
            "topic": "Riemann Hypothesis",
            "known_exhaustion_bounds": "已计算前 10^13 个非平凡零点，实部全部严格等于 1/2。",
            "implication": "寻找数值反例极度困难，不建议在此阶段浪费算力盲目搜零点。"
        }
    return {
        "topic": topic,
        "known_exhaustion_bounds": "暂无记录明确的计算机穷举边界，但请遵循一般原则：优先依靠解析与代数结构推导，而非暴力扫表。",
        "implication": "可以小范围探测规律，但大规模扫描前需证明其结构必要性。"
    }

def reverse_search_archive(
    memory_store: MemoryStore,
    query: str,
    categories: list[str] | None = None,
    limit: int = 10,
    run_id: str | None = None,
) -> dict[str, Any]:
    matches = memory_store.search_entries(query=query, categories=categories, limit=limit, run_id=run_id)
    return {"query": query, "matches": matches}


def list_recent_archive(memory_store: MemoryStore, run_id: str, limit: int = 20) -> dict[str, Any]:
    return {"run_id": run_id, "entries": memory_store.recent_entries(run_id=run_id, limit=limit)}


def run_summary(memory_store: MemoryStore, run_id: str) -> dict[str, Any]:
    return {"run": memory_store.get_run(run_id), "latest_checkpoint": memory_store.latest_checkpoint(run_id)}


def system_profile(resource_manager: ResourceManager) -> dict[str, Any]:
    snap = resource_manager.snapshot()
    return {
        "snapshot": snap.to_dict(),
        "recommended_batch_workers": resource_manager.recommend_batch_workers(),
    }


def beal_analyze_tool(x: int, y: int, z: int, moduli: list[int] | None = None) -> dict[str, Any]:
    moduli = tuple(moduli or (8, 9, 16, 25, 27, 32, 49))
    return analyze_exponent_triple(x=x, y=y, z=z, moduli=moduli)


def beal_recommend_tool(
    exp_min: int = 3,
    exp_max: int = 12,
    moduli: list[int] | None = None,
    mode: str = "counterexample",
    top: int = 15,
    symmetric: bool = True,
) -> dict[str, Any]:
    moduli = tuple(moduli or (8, 9, 16, 25, 27, 32, 49))
    return {
        "mode": mode,
        "recommendations": recommend_triples(
            exp_min=exp_min,
            exp_max=exp_max,
            moduli=moduli,
            mode=mode,
            top=top,
            symmetric=symmetric,
        ),
    }


def build_default_registry(memory_store: MemoryStore, use_sympy_tools: bool = True) -> ToolRegistry:
    registry = ToolRegistry()
    
    # 核心逻辑与证明管理工具
    registry.register("proof_tree_manager", "管理数学证明的逻辑结构树（添加引理、更新状态、查看依赖）。", proof_tree_manager)
    registry.register("retrieve_known_mathematical_bounds", "查询某个数学猜想目前被人类用计算机穷举验证到了什么边界，用于避免重复算号。", retrieve_known_mathematical_bounds)
    
    # 基础数论分析（仅保留轻量级分析，移除暴力搜索）
    registry.register("gcd_lcm", "计算最大公约数和最小公倍数。", gcd_lcm)
    registry.register("prime_factorization", "对整数做素因子分解。", prime_factorization)
    registry.register("is_prime", "判断整数是否为素数。", is_prime)
    registry.register("divisors", "列出整数的全部因子。", divisors)
    registry.register("p_adic_valuation", "计算整数在素数 p 下的 p-adic 估值。", p_adic_valuation)
    registry.register("modular_power", "计算模幂。", modular_power)
    registry.register("residue_pair_filter", "分析 A^x + B^y = C^z 在给定模数下的剩余对存活情况。参数: x(int), y(int), z(int), modulus(int)", residue_pair_filter)
    registry.register("beal_analyze", "分析比尔猜想指数三元组的模约束强度。", beal_analyze_tool)
    registry.register("beal_recommend", "推荐优先研究的比尔猜想指数三元组。", beal_recommend_tool)

    # 符号推导与形式化验证 (SymPy) - 证明的核心武器
    registry.register("sympy_available", "检查当前环境是否可用 SymPy。", sympy_available)
    if use_sympy_tools and _has_sympy():
        registry.register("numeric_eval", "对符号表达式做带替换的数值求值。", numeric_eval)
        registry.register("sympy_simplify", "使用 SymPy 化简表达式。", sympy_simplify)
        registry.register("sympy_factor", "使用 SymPy 因式分解表达式。", sympy_factor)
        registry.register("sympy_expand", "使用 SymPy 展开表达式。", sympy_expand)
        registry.register("sympy_solve", "使用 SymPy 求解符号方程。", sympy_solve)
        registry.register("sympy_diophantine", "使用 SymPy 求解丢番图(不定)方程，用于寻找反例的结构限制。", sympy_diophantine)
        registry.register("sympy_groebner", "使用 SymPy 计算 Groebner 基，用于多项式理想的成员问题。", sympy_groebner)
        registry.register("sympy_is_identity", "使用 SymPy 检查两个代数表达式是否恒等。", sympy_is_identity)
        registry.register("sympy_diff", "使用 SymPy 求导。", sympy_diff)
        registry.register("sympy_integrate", "使用 SymPy 积分。", sympy_integrate)
        registry.register("sympy_limit", "使用 SymPy 计算极限。", sympy_limit)
        registry.register("sympy_series", "使用 SymPy 展开级数。", sympy_series)
        registry.register("sympy_matrix_det", "使用 SymPy 计算矩阵行列式。", sympy_matrix_det)
        registry.register("sympy_matrix_eigenvals", "使用 SymPy 计算矩阵特征值。", sympy_matrix_eigenvals)
        registry.register("sympy_logic_prove", "验证一个布尔逻辑命题是否为永真式。参数: proposition(str)", sympy_logic_prove)
        registry.register("z3_logic_prove", "使用 Z3 SMT 求解器验证逻辑命题。参数: proposition(str) 例如 'ForAll([x,y], x+y == y+x)'", z3_logic_prove)
    registry.register(
        "reverse_search_archive",
        "逆向检索记忆库中的失败路径、启发式和历史结论。",
        lambda **kwargs: reverse_search_archive(memory_store=memory_store, **kwargs),
    )
    registry.register(
        "list_recent_archive",
        "读取某次研究最近的归档记忆。",
        lambda **kwargs: list_recent_archive(memory_store=memory_store, **kwargs),
    )
    registry.register(
        "run_summary",
        "读取某次研究的运行摘要和最新检查点。",
        lambda **kwargs: run_summary(memory_store=memory_store, **kwargs),
    )
    resource_manager = ResourceManager()
    registry.register("system_profile", "读取当前机器的 CPU、内存与推荐批处理并行度。", lambda **kwargs: system_profile(resource_manager=resource_manager))
    registry.register("retrieve_known_mathematical_bounds", "查询某个数学猜想目前被人类用计算机穷举验证到了什么边界，用于避免重复算号。", retrieve_known_mathematical_bounds)
    return registry
