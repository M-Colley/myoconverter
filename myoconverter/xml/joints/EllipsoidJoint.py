""" Contains the `EllipsoidJoint` parser.

Approximates an OpenSim 4.x `EllipsoidJoint` (used in scapulothoracic / shoulder
and wheelchair SimTK models) with three sequential rotational `hinge` joints
preserving the OpenSim Coordinate names, bounds, and locks.
"""

from loguru import logger

from myoconverter.xml.joints.BallJoint import BallJoint
from myoconverter.xml.utils import str2vec


class EllipsoidJoint(BallJoint):
  """ This class parses and converts the OpenSim `EllipsoidJoint` XML element to MuJoCo. """

  def _parse(self, xml, socket_parent_frame, socket_child_frame, pointer):
    radii_el = xml.find("radii_x_y_z")
    if radii_el is not None and radii_el.text is not None:
      radii = str2vec(radii_el.text)
      logger.warning(
        f"Approximating EllipsoidJoint '{xml.attrib.get('name', '')}' "
        f"(radii_x_y_z={radii.tolist()}) as a 3-DoF spherical hinge sequence in MuJoCo."
      )
    return super()._parse(xml, socket_parent_frame, socket_child_frame, pointer)
