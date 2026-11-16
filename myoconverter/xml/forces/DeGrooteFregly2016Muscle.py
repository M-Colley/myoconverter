""" Contains the `DeGrooteFregly2016Muscle` parser.

Converts OpenSim 4.x / OpenSim Moco `DeGrooteFregly2016Muscle` actuators
(widely used in SimTK clinical gait, cerebral palsy `SimCP`, stroke, and
AddBiomechanics optimal-control models) to MuJoCo.
"""

import numpy as np

from myoconverter.xml.utils import element_txt2num
from myoconverter.xml.forces.Muscle import Muscle


class DeGrooteFregly2016Muscle(Muscle):
  """ This class parses and converts the OpenSim `DeGrooteFregly2016Muscle` XML element to MuJoCo. """

  def _parse(self, xml, **kwargs):
    activation1 = element_txt2num(xml, "activation_time_constant")
    activation2 = element_txt2num(xml, "deactivation_time_constant")
    activation1 = 0.015 if np.isnan(activation1) else activation1
    activation2 = 0.060 if np.isnan(activation2) else activation2
    self._params["timeconst"] = np.array([activation1, activation2])
