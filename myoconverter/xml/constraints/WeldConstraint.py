""" Contains the `WeldConstraint` parser.

Converts an OpenSim 4.x `WeldConstraint` (used for prosthetic socket attachment,
arthrodesis / joint fusions, surgical fixation, and closed kinematic chains such
as wheelchair hand-rim coupling) into a MuJoCo `<equality><weld .../></equality>`.
"""

from lxml import etree
import numpy as np
from scipy.spatial.transform import Rotation
from loguru import logger

from myoconverter.xml.parsers import IParser
from myoconverter.xml.utils import (
  create_transformation_matrix,
  get_body_and_transform,
  str2bool,
  str2vec,
  vec2str,
)
from myoconverter.xml import config as cfg


class WeldConstraint(IParser):
  """ This class parses and converts the OpenSim `WeldConstraint` XML element to MuJoCo. """

  def parse(self, xml):
    """ Schedule or execute `WeldConstraint` conversion once bodies exist in `M_WORLDBODY`. """
    if len(cfg.M_WORLDBODY.findall(".//body")) <= 1 and hasattr(cfg, "DEFERRED_CONSTRAINTS"):
      cfg.DEFERRED_CONSTRAINTS.append(lambda x=xml: self._materialize(x))
    else:
      self._materialize(xml)

  @staticmethod
  def _resolve_constraint_frame(xml, socket_tags, loc_tag, orient_tag):
    ref_text = None
    for tag in socket_tags:
      el = xml.find(tag)
      if el is not None and el.text is not None and el.text.strip():
        ref_text = el.text.strip()
        break
    if not ref_text:
      return None, None

    # Check if ref_text names a local PhysicalOffsetFrame inside <WeldConstraint><frames>
    leaf = [tok for tok in ref_text.split("/") if tok][-1]
    local_frame = xml.find(f"frames/*[@name='{leaf}']")
    if local_frame is not None and local_frame.tag == "PhysicalOffsetFrame":
      sp = local_frame.find("socket_parent")
      parent_ref = sp.text.strip() if sp is not None and sp.text else "ground"
      body, T_parent = get_body_and_transform(cfg.OPENSIM, cfg.M_WORLDBODY, parent_ref)
      p_el = local_frame.find("translation")
      o_el = local_frame.find("orientation")
      p = str2vec(p_el.text) if p_el is not None and p_el.text else np.zeros(3)
      eul = str2vec(o_el.text) if o_el is not None and o_el.text else np.zeros(3)
      return body, np.matmul(T_parent, create_transformation_matrix(pos=p, euler=eul))

    body, T_frame = get_body_and_transform(cfg.OPENSIM, cfg.M_WORLDBODY, ref_text)
    loc_el = xml.find(loc_tag) if loc_tag else None
    ori_el = xml.find(orient_tag) if orient_tag else None
    if (loc_el is not None and loc_el.text) or (ori_el is not None and ori_el.text):
      p = str2vec(loc_el.text) if loc_el is not None and loc_el.text else np.zeros(3)
      eul = str2vec(ori_el.text) if ori_el is not None and ori_el.text else np.zeros(3)
      T_frame = np.matmul(T_frame, create_transformation_matrix(pos=p, euler=eul))
    return body, T_frame

  def _materialize(self, xml):
    name = xml.attrib.get("name", "weld_constraint")
    try:
      body1, T_b1_f1 = self._resolve_constraint_frame(
        xml, ("socket_frame1", "body_1"), "location_body_1", "orientation_body_1"
      )
      body2, T_b2_f2 = self._resolve_constraint_frame(
        xml, ("socket_frame2", "body_2"), "location_body_2", "orientation_body_2"
      )
    except RuntimeError:
      logger.warning(f"Could not resolve frames for WeldConstraint '{name}', skipping.")
      return

    if body1 is None or body2 is None:
      logger.warning(f"WeldConstraint '{name}' is missing frame references, skipping.")
      return

    # T_{B1, B2} = T_{B1, F1} * inv(T_{B2, F2})
    T_b1_b2 = np.matmul(T_b1_f1, np.linalg.inv(T_b2_f2))
    rel_pos = T_b1_b2[:3, 3]
    xyzw = Rotation.from_matrix(T_b1_b2[:3, :3]).as_quat()
    wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=float)
    relpose = np.concatenate([rel_pos, wxyz])

    is_enforced = xml.find("isEnforced")
    active = "true" if is_enforced is None or not is_enforced.text else (
      "true" if str2bool(is_enforced.text.strip()) else "false"
    )

    b1_name = body1.attrib.get("name", "world")
    b2_name = body2.attrib.get("name", "world")
    attrib = {
      "name": name,
      "body1": b1_name,
      "body2": b2_name,
      "anchor": "0 0 0",
      "relpose": vec2str(relpose),
      "active": active,
      "solimp": "0.9999 0.9999 0.001 0.5 2",
    }
    etree.SubElement(cfg.M_EQUALITY, "weld", attrib=attrib)
