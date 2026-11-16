""" Contains the `ExpressionBasedCoordinateForce` parser.

Safely evaluates OpenSim 4.x `ExpressionBasedCoordinateForce` mathematical
expressions (via a strict AST whitelist, never `eval`) over the target
coordinate's operating range and fits passive joint stiffness / springref
(& viscous damping when `qdot` appears linearly).
"""

import ast
import operator
import numpy as np
from loguru import logger

from myoconverter.xml import config as cfg
from myoconverter.xml.parsers import IParser
from myoconverter.xml.utils import str2bool, str2vec

_SAFE_FUNCS = {
  "exp": np.exp,
  "sin": np.sin,
  "cos": np.cos,
  "tan": np.tan,
  "asin": np.arcsin,
  "acos": np.arccos,
  "atan": np.arctan,
  "sinh": np.sinh,
  "cosh": np.cosh,
  "tanh": np.tanh,
  "abs": np.abs,
  "sqrt": np.sqrt,
  "log": np.log,
  "log10": np.log10,
  "min": np.minimum,
  "max": np.maximum,
  "pow": np.power,
  "step": lambda x: np.where(x >= 0.0, 1.0, 0.0),
}

_BIN_OPS = {
  ast.Add: operator.add,
  ast.Sub: operator.sub,
  ast.Mult: operator.mul,
  ast.Div: operator.truediv,
  ast.Pow: operator.pow,
}

_UNARY_OPS = {
  ast.UAdd: operator.pos,
  ast.USub: operator.neg,
}


def _eval_math_ast(node, env):
  if isinstance(node, ast.Expression):
    return _eval_math_ast(node.body, env)
  if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
    return float(node.value)
  if isinstance(node, ast.Name):
    if node.id in env:
      return env[node.id]
    if node.id == "pi":
      return float(np.pi)
    if node.id == "e":
      return float(np.e)
    raise ValueError(f"Unsupported identifier '{node.id}' in OpenSim expression")
  if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
    return _UNARY_OPS[type(node.op)](_eval_math_ast(node.operand, env))
  if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
    return _BIN_OPS[type(node.op)](_eval_math_ast(node.left, env), _eval_math_ast(node.right, env))
  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCS:
    args = [_eval_math_ast(arg, env) for arg in node.args]
    return _SAFE_FUNCS[node.func.id](*args)
  raise ValueError(f"Unsupported AST construct {type(node).__name__} in OpenSim expression")


class ExpressionBasedCoordinateForce(IParser):
  """ This class parses and converts the OpenSim `ExpressionBasedCoordinateForce` XML element to MuJoCo. """

  def parse(self, xml):
    if xml.find("appliesForce") is not None and not str2bool(xml.find("appliesForce").text):
      return

    name = xml.attrib.get("name", "expression_force")
    coord_el = xml.find("coordinate")
    expr_el = xml.find("expression")
    if coord_el is None or not coord_el.text or expr_el is None or not expr_el.text:
      logger.warning(f"ExpressionBasedCoordinateForce '{name}' missing coordinate or expression, skipping.")
      return

    coord_name = coord_el.text.strip()
    joint = cfg.M_WORLDBODY.find(f".//joint[@name='{coord_name}']")
    if joint is None:
      logger.warning(
        f"Joint '{coord_name}' not found for ExpressionBasedCoordinateForce '{name}', skipping."
      )
      return

    # OpenSim Lepton expressions use `^` for exponentiation and allow whitespace removal
    expr_src = expr_el.text.strip().replace("^", "**")
    try:
      parsed = ast.parse(expr_src, mode="eval")
    except SyntaxError as exc:
      logger.warning(f"Could not parse expression on '{name}' ({exc}), skipping.")
      return

    if "range" in joint.attrib:
      q_min, q_max = str2vec(joint.attrib["range"])
      if np.isclose(q_min, q_max):
        q_min, q_max = q_min - 0.25, q_max + 0.25
    else:
      q_min, q_max = -0.5, 0.5

    q_grid = np.linspace(float(q_min), float(q_max), 65)
    try:
      f_q = np.broadcast_to(
        np.asarray(_eval_math_ast(parsed, {"q": q_grid, "qdot": np.zeros_like(q_grid)}), dtype=float),
        q_grid.shape,
      )
      f_qd1 = np.broadcast_to(
        np.asarray(_eval_math_ast(parsed, {"q": np.zeros_like(q_grid), "qdot": np.ones_like(q_grid)}), dtype=float),
        q_grid.shape,
      )
      f_qd0 = np.broadcast_to(
        np.asarray(_eval_math_ast(parsed, {"q": np.zeros_like(q_grid), "qdot": np.zeros_like(q_grid)}), dtype=float),
        q_grid.shape,
      )
    except Exception as exc:
      logger.warning(f"Could not evaluate ExpressionBasedCoordinateForce '{name}' ({exc}), skipping.")
      return

    if not np.all(np.isfinite(f_q)):
      logger.warning(f"Non-finite values in ExpressionBasedCoordinateForce '{name}', skipping.")
      return

    # Linear fit F(q) ≈ slope * q + intercept  =>  passive restoring stiffness k = -slope, springref q0 = intercept / k
    slope, intercept = np.polyfit(q_grid, f_q, 1)
    f_hat = slope * q_grid + intercept
    ss_tot = float(np.sum((f_q - np.mean(f_q)) ** 2))
    r2 = 1.0 if ss_tot < 1e-12 else 1.0 - float(np.sum((f_q - f_hat) ** 2)) / ss_tot
    if r2 < 0.85:
      logger.warning(
        f"Nonlinear ExpressionBasedCoordinateForce '{name}' on '{coord_name}' linearized with R^2={r2:.3f}."
      )

    k_add = max(0.0, float(-slope))
    if k_add > 1e-8:
      q0_add = float(intercept / k_add)
      k_prev = float(joint.attrib.get("stiffness", "0"))
      q0_prev = float(joint.attrib.get("springref", "0"))
      k_total = k_prev + k_add
      q0_total = (k_prev * q0_prev + k_add * q0_add) / k_total
      joint.attrib["stiffness"] = f"{k_total:.6g}"
      joint.attrib["springref"] = f"{q0_total:.6g}"

    c_add = max(0.0, float(-np.mean(f_qd1 - f_qd0)))
    if c_add > 1e-8:
      c_prev = float(joint.attrib.get("damping", "0"))
      joint.attrib["damping"] = f"{c_prev + c_add:.6g}"
