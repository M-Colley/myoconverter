""" This module contains a collection of utility functions useful for parsing and converting the OpenSim `ForceSet`

Muscle conversion analysis by Florian Fischer and Miroslav Bachinski.

@author: Aleksi Ikkala
"""

import numpy as np
from scipy.optimize import minimize


def mujoco_LO_loss(length_range, range, optimal_fiber_length, tendon_slack_length, pennation_angle):
  """
  Computes squared Euclidean distance between MuJoCo and OpenSim model,
  regarding both optimal fiber length and constant tendon length/tendon slack length.

  Original code for this function was provided by Florian Fischer (2022).

  :param length_range: array of MuJoCo tendon length (=complete actuator length) ranges
  :param range: Operating length of muscle
  :param optimal_fiber_length: OpenSim optimal fiber length
  :param tendon_slack_length: OpenSim tendon slack length (or any reasonable constant tendon lengths)
  :param pennation_angle: OpenSim pennation angle at optimum
          (i.e., angle between tendon and fibers at optimal fiber length expressed in radians)
  :param use_optPennationAngle: Boolean; if this set to True, MuJoCo optimal fiber lengths LO should match
          OpenSim optimal fiber lengths LO_osim * cos(OpenSim pennation angle at optimum); otherwise, LO should match LO_osim
  :return: squared (unweighted) Euclidean distance of optimal fiber length and constant tendon lengths between MuJoCo and OpenSim
  """
  LO = estimate_fiber_length(length_range, range)
  LT = estimate_tendon_slack_length(length_range, range)

  if np.isnan(pennation_angle):
    pennation_angle = 0

  return np.linalg.norm(LO - optimal_fiber_length * np.cos(pennation_angle)) ** 2 + np.linalg.norm(
    LT - tendon_slack_length) ** 2


def estimate_fiber_length(length_range, range):
  """ Code by Florian Fischer """
  return (length_range[0] - length_range[1]) / (range[0] - range[1])


def estimate_tendon_slack_length(length_range, range):
  """ Code by Florian Fischer """
  return length_range[0] - range[0] * estimate_fiber_length(length_range, range)


def calculate_length_range(range, optimal_fiber_length, tendon_slack_length, pennation_angle):
  """ Length range computations by Florian Fischer """

  # Estimate length range if optimal fiber length and tendon slack length are defined
  if np.all(np.isfinite([optimal_fiber_length, tendon_slack_length, pennation_angle])):

    # Estimate actuator length ranges by minimizing error between mujoco and opensim optimal fiber length
    length_range = np.array([0.5, 2]) * tendon_slack_length
    sol = minimize(mujoco_LO_loss, length_range,
                   args=(range, optimal_fiber_length, tendon_slack_length, pennation_angle))

    if sol.success:
      return sol.x

  return np.array([np.nan, np.nan])


def ensure_contact_geometry(geom_ref):
  """Materialize an OpenSim `<ContactSphere>` or `<ContactHalfSpace>` as a MuJoCo `<geom>` and return its name."""
  from lxml import etree
  from loguru import logger
  from scipy.spatial.transform import Rotation

  from myoconverter.xml import config as cfg
  from myoconverter.xml.utils import element_txt2num, get_body_and_transform, str2vec, vec2str

  if not geom_ref:
    return None
  leaf = [tok for tok in geom_ref.strip().split("/") if tok][-1]

  existing = cfg.M_WORLDBODY.find(f".//geom[@name='{leaf}']")
  if existing is not None:
    return leaf

  osim_geom = None
  for item in getattr(cfg, "O_CONTACTGEOMETRYSET", []):
    if item.attrib.get("name") == leaf:
      osim_geom = item
      break
  if osim_geom is None and cfg.OPENSIM is not None:
    for tag in ("ContactSphere", "ContactHalfSpace", "ContactMesh"):
      osim_geom = cfg.OPENSIM.find(f".//{tag}[@name='{leaf}']")
      if osim_geom is not None:
        break

  if osim_geom is None:
    logger.warning(f"Referenced OpenSim ContactGeometry '{geom_ref}' not found, skipping.")
    return None

  frame_ref = None
  for ftag in ("socket_frame", "body_name", "body"):
    el = osim_geom.find(ftag)
    if el is not None and el.text is not None and el.text.strip():
      frame_ref = el.text.strip()
      break
  if not frame_ref:
    frame_ref = "ground"

  try:
    parent_body, T_frame = get_body_and_transform(cfg.OPENSIM, cfg.M_WORLDBODY, frame_ref)
  except RuntimeError:
    logger.warning(f"Could not resolve parent body '{frame_ref}' for ContactGeometry '{leaf}', skipping.")
    return None

  loc_el = osim_geom.find("location")
  loc = str2vec(loc_el.text) if loc_el is not None and loc_el.text else np.zeros(3)
  pos_in_body = (T_frame @ np.append(loc, 1.0))[:3]

  ori_el = osim_geom.find("orientation")
  ori = str2vec(ori_el.text) if ori_el is not None and ori_el.text else np.zeros(3)
  R_geom = T_frame[:3, :3] @ Rotation.from_euler("XYZ", ori).as_matrix()

  if osim_geom.tag == "ContactSphere":
    radius = element_txt2num(osim_geom, "radius", default=0.01)
    etree.SubElement(
      parent_body,
      "geom",
      name=leaf,
      type="sphere",
      pos=vec2str(pos_in_body),
      size=f"{max(float(radius), 1e-5):.6g}",
      rgba="0.2 0.8 0.4 0.5",
      group="3",
    )
    return leaf

  if osim_geom.tag == "ContactHalfSpace":
    # OpenSim ContactHalfSpace occupies +X; outward free-space normal in local frame is [-1, 0, 0]
    outward_normal = R_geom @ np.array([-1.0, 0.0, 0.0])
    norm = np.linalg.norm(outward_normal)
    if norm > 0:
      outward_normal = outward_normal / norm
    else:
      outward_normal = np.array([0.0, 1.0, 0.0])
    etree.SubElement(
      parent_body,
      "geom",
      name=leaf,
      type="plane",
      pos=vec2str(pos_in_body),
      zaxis=vec2str(outward_normal),
      size="10 10 0.05",
      rgba="0.85 0.85 0.85 0.3",
      group="3",
    )
    return leaf

  logger.warning(f"Unsupported ContactGeometry type <{osim_geom.tag} name='{leaf}'>, skipping.")
  return None
