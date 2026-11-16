""" Contains the `FreeJoint` parser.

Converts an OpenSim 4.x `FreeJoint` (6 Coordinates: `[rx, ry, rz, tx, ty, tz]`)
into three MuJoCo `slide` joints followed by three `hinge` joints, preserving
exact OpenSim Coordinate names, bounds, locks, and Step 1-3 coordinate indexing
while remaining MJX-compatible.
"""

from copy import deepcopy
from lxml import etree
import numpy as np
from loguru import logger

from myoconverter.xml.joints.Joint import Joint
from myoconverter.xml.joints.utils import parse_coordinates, estimate_axis
from myoconverter.xml.utils import val2str, filter_keys


class FreeJoint(Joint):
  """ This class parses and converts the OpenSim `FreeJoint` XML element to MuJoCo. """

  def _parse(self, xml, socket_parent_frame, socket_child_frame, pointer):
    """ Parse an OpenSim `FreeJoint` into 3 slide + 3 hinge MuJoCo joints.

    :param xml: OpenSim `FreeJoint` XML element
    :param socket_parent_frame: Parent frame socket
    :param socket_child_frame: Child frame socket
    :param pointer: Pointer to the MuJoCo XML body where joints will be added
    :return: A list of MuJoCo XML joints, a list of joint parameters
    :raises RuntimeError: If the number of defined Coordinates is not 6
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
        for i in range(6)
      }
    elif len(coordinates) != 6:
      logger.critical(
        f"There should be six Coordinates for a FreeJoint, but {joint_name} has {len(coordinates)}"
      )
      raise RuntimeError

    coord_list = list(coordinates.values())
    basis_axes = (
      np.array([1.0, 0.0, 0.0]),
      np.array([0.0, 1.0, 0.0]),
      np.array([0.0, 0.0, 1.0]),
    )

    # Emit translations [3, 4, 5] before rotations [0, 1, 2] so translational axes stay in parent frame
    spec = [
      (3, "slide", basis_axes[0]),
      (4, "slide", basis_axes[1]),
      (5, "slide", basis_axes[2]),
      (0, "hinge", basis_axes[0]),
      (1, "hinge", basis_axes[1]),
      (2, "hinge", basis_axes[2]),
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
