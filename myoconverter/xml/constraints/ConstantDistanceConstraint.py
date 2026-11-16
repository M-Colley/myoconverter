""" Contains the `ConstantDistanceConstraint` parser.

Converts an OpenSim 4.x `ConstantDistanceConstraint` (rigid-rod distance link
used in surgical fixation and prosthetic linkages) into a 2-site MuJoCo spatial
tendon with an equality distance constraint.
"""

from lxml import etree
import numpy as np
from loguru import logger

from myoconverter.xml.parsers import IParser
from myoconverter.xml.utils import (
  element_txt2num,
  get_body_and_transform,
  str2bool,
  str2vec,
  vec2str,
)
from myoconverter.xml import config as cfg


class ConstantDistanceConstraint(IParser):
  """ This class parses and converts the OpenSim `ConstantDistanceConstraint` XML element to MuJoCo. """

  def parse(self, xml):
    """ Schedule or execute `ConstantDistanceConstraint` conversion once bodies exist in `M_WORLDBODY`. """
    if len(cfg.M_WORLDBODY.findall(".//body")) <= 1 and hasattr(cfg, "DEFERRED_CONSTRAINTS"):
      cfg.DEFERRED_CONSTRAINTS.append(lambda x=xml: self._materialize(x))
    else:
      self._materialize(xml)

  @staticmethod
  def _read_ref(xml, tags):
    for tag in tags:
      el = xml.find(tag)
      if el is not None and el.text is not None and el.text.strip():
        return el.text.strip()
    return None

  def _materialize(self, xml):
    name = xml.attrib.get("name", "constant_distance_constraint")
    b1_ref = self._read_ref(xml, ("socket_body_1", "body_1"))
    b2_ref = self._read_ref(xml, ("socket_body_2", "body_2"))
    if not b1_ref or not b2_ref:
      logger.warning(f"ConstantDistanceConstraint '{name}' is missing body references, skipping.")
      return

    try:
      body1, T_f1 = get_body_and_transform(cfg.OPENSIM, cfg.M_WORLDBODY, b1_ref)
      body2, T_f2 = get_body_and_transform(cfg.OPENSIM, cfg.M_WORLDBODY, b2_ref)
    except RuntimeError:
      logger.warning(f"Could not resolve bodies for ConstantDistanceConstraint '{name}', skipping.")
      return

    loc1_el = xml.find("location_body_1")
    loc2_el = xml.find("location_body_2")
    loc1 = str2vec(loc1_el.text) if loc1_el is not None and loc1_el.text else np.zeros(3)
    loc2 = str2vec(loc2_el.text) if loc2_el is not None and loc2_el.text else np.zeros(3)
    p1 = (T_f1 @ np.append(loc1, 1.0))[:3]
    p2 = (T_f2 @ np.append(loc2, 1.0))[:3]

    dist = element_txt2num(xml, "constant_distance", default=0.0)
    if not np.isfinite(dist) or dist < 0.0:
      logger.warning(f"Invalid constant_distance on '{name}', skipping.")
      return

    site1_name = f"{name}_dist_site1"
    site2_name = f"{name}_dist_site2"
    tendon_name = f"{name}_dist_tendon"
    etree.SubElement(body1, "site", name=site1_name, pos=vec2str(p1))
    etree.SubElement(body2, "site", name=site2_name, pos=vec2str(p2))

    spatial = etree.SubElement(cfg.M_TENDON, "spatial", name=tendon_name)
    etree.SubElement(spatial, "site", site=site1_name)
    etree.SubElement(spatial, "site", site=site2_name)

    is_enforced = xml.find("isEnforced")
    active = "true" if is_enforced is None or not is_enforced.text else (
      "true" if str2bool(is_enforced.text.strip()) else "false"
    )

    etree.SubElement(
      cfg.M_EQUALITY,
      "tendon",
      name=name,
      tendon1=tendon_name,
      polycoef=f"{float(dist):.6g} 0 0 0 0",
      active=active,
      solimp="0.9999 0.9999 0.001 0.5 2",
    )
