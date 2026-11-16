""" Contains the `PlanarJoint` parser.

Converts an OpenSim 4.x `PlanarJoint` (3 Coordinates: `[rz, tx, ty]`) into two
MuJoCo `slide` joints (`tx`, `ty`) followed by one `hinge` joint (`rz`).
Commonly used in 2D clinical gait, prosthetic, and non-human SimTK models.
"""

from copy import deepcopy
from lxml import etree
import numpy as np
from loguru import logger

from myoconverter.xml.joints.Joint import Joint
from myoconverter.xml.joints.utils import parse_coordinates, estimate_axis
from myoconverter.xml.utils import val2str, filter_keys


class PlanarJoint(Joint):
  """ This class parses and converts the OpenSim `PlanarJoint` XML element to MuJoCo. """

  def _parse(self, xml, socket_parent_frame, socket_child_frame, pointer):
    """ Parse an OpenSim `PlanarJoint` into two slide joints and one hinge joint.

    :param xml: OpenSim `PlanarJoint` XML element
    :param socket_parent_frame: Parent frame socket
    :param socket_child_frame: Child frame socket
    :param pointer: Pointer to the MuJoCo XML body where joints will be added
    :return: A list of MuJoCo XML joints, a list of joint parameters
    :raises RuntimeError: If the number of defined Coordinates is not 3
    """
    joint_name = xml.attrib["name"]
    coordinates = parse_coordinates(xml.find("coordinates"))
    if not coordinates:
      coordinates = {
        f"{joint_name}_coord_{i}": {
          "name": f"{joint_name}_coord_{i}",
          "limited": False,
          "_locked": False,
          "user": 0.0,
          "_transform_value": 0.0,
        }
        for i in range(3)
      }
    elif len(coordinates) != 3:
      logger.critical(
        f"There should be three Coordinates for a PlanarJoint, but {joint_name} has {len(coordinates)}"
      )
      raise RuntimeError

    coord_list = list(coordinates.values())
    # OpenSim PlanarJoint order: [rz, tx, ty]; emit translations before in-plane rotation
    spec = [
      (1, "slide", np.array([1.0, 0.0, 0.0])),
      (2, "slide", np.array([0.0, 1.0, 0.0])),
      (0, "hinge", np.array([0.0, 0.0, 1.0])),
    ]

    all_joints = []
    all_params = []
    for idx, jtype, axis_vec in spec:
      params = deepcopy(coord_list[idx])
      params["ref"] = 0
      params["type"] = jtype
      params["axis"] = estimate_axis(socket_child_frame, axis_vec)
      all_joints.append(etree.SubElement(pointer, "joint", attrib=val2str(filter_keys(params))))
      all_params.append(params)

    return all_joints, all_params
