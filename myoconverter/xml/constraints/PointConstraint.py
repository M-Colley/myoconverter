""" Contains the `PointConstraint` parser.

Converts an OpenSim 4.x `PointConstraint` (spherical / ball-socket closed-loop
constraint used in four-bar prosthetic knees, scapulothoracic linkages, and
surgical models) into a MuJoCo `<equality><connect .../></equality>`.
"""

from lxml import etree
import numpy as np
from loguru import logger

from myoconverter.xml.parsers import IParser
from myoconverter.xml.utils import get_body_and_transform, str2bool, str2vec, vec2str
from myoconverter.xml import config as cfg


class PointConstraint(IParser):
  """ This class parses and converts the OpenSim `PointConstraint` XML element to MuJoCo. """

  def parse(self, xml):
    """ Schedule or execute `PointConstraint` conversion once bodies exist in `M_WORLDBODY`. """
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
    name = xml.attrib.get("name", "point_constraint")
    b1_ref = self._read_ref(xml, ("socket_body_1", "body_1"))
    b2_ref = self._read_ref(xml, ("socket_body_2", "body_2"))
    if not b1_ref or not b2_ref:
      logger.warning(f"PointConstraint '{name}' is missing body references, skipping.")
      return

    try:
      body1, T_f1 = get_body_and_transform(cfg.OPENSIM, cfg.M_WORLDBODY, b1_ref)
      body2, _ = get_body_and_transform(cfg.OPENSIM, cfg.M_WORLDBODY, b2_ref)
    except RuntimeError:
      logger.warning(f"Could not resolve bodies for PointConstraint '{name}', skipping.")
      return

    loc1_el = xml.find("location_body_1")
    loc1 = str2vec(loc1_el.text) if loc1_el is not None and loc1_el.text else np.zeros(3)
    anchor_in_b1 = (T_f1 @ np.append(loc1, 1.0))[:3]

    is_enforced = xml.find("isEnforced")
    active = "true" if is_enforced is None or not is_enforced.text else (
      "true" if str2bool(is_enforced.text.strip()) else "false"
    )

    etree.SubElement(
      cfg.M_EQUALITY,
      "connect",
      name=name,
      body1=body1.attrib.get("name", "ground"),
      body2=body2.attrib.get("name", "ground"),
      anchor=vec2str(anchor_in_b1),
      active=active,
      solimp="0.9999 0.9999 0.001 0.5 2",
    )
