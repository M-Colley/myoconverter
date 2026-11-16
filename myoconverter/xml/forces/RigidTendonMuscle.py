""" Contains the `RigidTendonMuscle` parser. """

import numpy as np

from myoconverter.xml.forces.Muscle import Muscle


class RigidTendonMuscle(Muscle):
  """ This class parses and converts the OpenSim `RigidTendonMuscle` XML element to MuJoCo. """

  def _parse(self, xml, **kwargs):
    self._params["timeconst"] = np.array([0.01, 0.04])
