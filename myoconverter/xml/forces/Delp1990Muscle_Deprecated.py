""" Contains the `Delp1990Muscle_Deprecated` parser.

Commonly encountered in classic SimTK surgical, tendon-transfer, wrist, and
neck musculoskeletal `.osim` models upgraded to OpenSim 4.x XML format.
"""

import numpy as np

from myoconverter.xml.utils import element_txt2num
from myoconverter.xml.forces.Muscle import Muscle


class Delp1990Muscle_Deprecated(Muscle):
  """ This class parses and converts the OpenSim `Delp1990Muscle_Deprecated` XML element to MuJoCo. """

  def _parse(self, xml, **kwargs):
    activation1 = element_txt2num(xml, "activation1", default=7.667)
    activation2 = element_txt2num(xml, "activation2", default=1.4549)
    time_scale = element_txt2num(xml, "time_scale", default=0.1)

    if time_scale <= 0 or np.isnan(activation1) or np.isnan(activation2):
      self._params["timeconst"] = np.array([0.015, 0.05])
    else:
      act_lin = ctrl_lin = 0.5
      denom_act = (0.5 + 1.5 * act_lin) * (activation1 * ctrl_lin + activation2)
      time_act = time_scale / max(denom_act, 1e-6)
      time_deact = time_scale * (0.5 + 1.5 * act_lin) / max(activation2, 1e-6)
      self._params["timeconst"] = np.array([time_act, time_deact])
