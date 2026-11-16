""" Contains the `ActivationCoordinateActuator` parser.

Converts OpenSim 4.x `ActivationCoordinateActuator` elements (used in prosthetic,
orthotic, exoskeleton, and clinical reserve-actuator `.osim` models) into MuJoCo
`<general>` actuators with first-order `filter` activation dynamics.
"""

from lxml import etree
import numpy as np
from loguru import logger

from myoconverter.xml import config as cfg
from myoconverter.xml.forces.TorqueActuator import TorqueActuator
from myoconverter.xml.parsers import IParser
from myoconverter.xml.utils import (
  element_txt2num,
  filter_keys,
  filter_nan_values,
  str2bool,
  val2str,
)


class ActivationCoordinateActuator(IParser):
  """ This class parses and converts the OpenSim `ActivationCoordinateActuator` XML element to MuJoCo. """

  def parse(self, xml):
    if xml.find("appliesForce") is not None and not str2bool(xml.find("appliesForce").text):
      return

    coord_el = xml.find("coordinate")
    if coord_el is None or not coord_el.text:
      logger.warning(f"ActivationCoordinateActuator '{xml.attrib.get('name', '')}' has no coordinate, skipping.")
      return

    coord_name = coord_el.text.strip()
    joint = cfg.M_WORLDBODY.find(f".//joint[@name='{coord_name}']")
    if joint is None:
      logger.warning(
        f"Joint '{coord_name}' was not found in the converted MuJoCo model for "
        f"ActivationCoordinateActuator '{xml.attrib.get('name', '')}', skipping."
      )
      return

    optimal_force = element_txt2num(xml, "optimal_force", default=1.0)
    if not np.isfinite(optimal_force):
      optimal_force = 1.0
    tau = element_txt2num(xml, "activation_time_constant", default=0.01)
    if not np.isfinite(tau) or tau <= 0.0:
      tau = 0.01

    params = {
      "name": xml.attrib["name"],
      "joint": coord_name,
      "dyntype": "filter",
      "dynprm": f"{float(tau):.6g} 0 0 0 0 0 0 0 0 0",
      "gainprm": f"{float(optimal_force):.6g} 0 0 0 0 0 0 0 0 0",
    }
    TorqueActuator._parse_control_limits(xml, params)

    first_muscle = cfg.M_ACTUATOR.find("muscle")
    idx = len(cfg.M_ACTUATOR) if first_muscle is None else cfg.M_ACTUATOR.index(first_muscle)
    cfg.M_ACTUATOR.insert(idx, etree.Element("general", attrib=val2str(filter_nan_values(filter_keys(params)))))
