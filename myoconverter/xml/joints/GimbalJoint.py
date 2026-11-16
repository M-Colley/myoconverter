""" Contains the `GimbalJoint` parser.

Converts an OpenSim 4.x `GimbalJoint` (3-DoF body-fixed XYZ Euler rotations)
into three sequential MuJoCo `hinge` joints.
"""

from myoconverter.xml.joints.BallJoint import BallJoint


class GimbalJoint(BallJoint):
  """ This class parses and converts the OpenSim `GimbalJoint` XML element to MuJoCo. """
