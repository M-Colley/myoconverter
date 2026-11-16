""" Contains the `Millard2012AccelerationMuscle` parser. """

import numpy as np

from myoconverter.xml.utils import element_txt2num
from myoconverter.xml.forces.Muscle import Muscle


class Millard2012AccelerationMuscle(Muscle):
  """ This class parses and converts the OpenSim `Millard2012AccelerationMuscle` XML element to MuJoCo. """

  def _parse(self, xml, **kwargs):
    activation1 = element_txt2num(xml, "activation_time_constant")
    activation2 = element_txt2num(xml, "deactivation_time_constant")
    activation1 = 0.01 if np.isnan(activation1) else activation1
    activation2 = 0.04 if np.isnan(activation2) else activation2
    self._params["timeconst"] = np.array([activation1, activation2])
