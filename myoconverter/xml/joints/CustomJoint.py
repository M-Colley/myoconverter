""" Contains the `CustomJoint` parser.

@author: Aleksi Ikkala
"""

from copy import deepcopy
import numpy as np
from lxml import etree
import os

from loguru import logger

from myoconverter.xml.joints.Joint import Joint
from myoconverter.xml.joints.utils import parse_coordinates, estimate_axis, plot_and_save_figure
from myoconverter.xml.utils import str2bool, str2vec, val2str, vec2str, filter_keys, fit_spline, is_linear
from myoconverter.xml import config as cfg


class CustomJoint(Joint):
  """ This class parses and converts the OpenSim `CustomJoint` XML element to MuJoCo.

  A CustomJoint in OpenSim can represent any type of a joint through 3 translational and 3 rotational degrees
  of freedom.
  """

  def _parse(self, xml, socket_parent_frame, socket_child_frame, pointer):
    """ This function handles the actual parsing and converting.

    The given CustomJoint is parsed into a set of MuJoCo joints (1-6 depending on how many non-zero non-constant
    joints there are).

    :param xml: OpenSim `CustomJoint` XML element
    :param socket_parent_frame: Parent frame socket
    :param socket_child_frame: Child frame socket
    :param pointer: A pointer to the MuJoCo XML file where this joint will be added
    :return: A list of MuJoCo XML joints, a list of joint parameters
    :raises: IndexError: If joint components are parsed in incorrect order
    :raises: TypeError: If a joint component type is not recognized
    :raises: RuntimeError: If spline definition cannot be found in the XML element
    :raises: NotImplementedError: If a joint component depends on two or more Coordinates
    :raises: NotImplementedError: If two Coordinates have a non-linear mapping
    :raises: RuntimeError: If a joint component has a non-identity linear mapping to a Coordinate
    :raises: RuntimeError: If a joint transformation type has not been implemented
    """

    # Return all XML definitions of joints and their params as lists
    all_joints = []
    all_params = []

    # Get transform axes
    transform_axes = xml.findall("SpatialTransform/TransformAxis")

    # Start by parsing the coordinates
    coordinates = parse_coordinates(xml.find("coordinates"))

    # NOTE! Coordinates in CoordinateSet parameterize this joint. In theory all six DoFs could be dependent
    # on one Coordinate. Here we assume that only one (MuJoCo) DoF is equivalent to a Coordinate, that is, there exists
    # an identity mapping between a Coordinate and a DoF, which is different to OpenSim where there might be no
    # identity mappings. In OpenSim a Coordinate is just a value and all DoFs might have some kind of mapping with
    # it, see e.g. "flexion" Coordinate in MoBL_ARMS_module6_7_CMC.osim model. MuJoCo doesn't have such abstract
    # notion of a "Coordinate", and thus there cannot be a non-identity mapping from a DoF to itself

    # Pre-check which coordinates in this joint have at least one unit-slope identity axis ([+/-1, 0])
    coords_with_unit_axis = set()
    for t_scan in transform_axes:
      c_el = t_scan.find("coordinates")
      if c_el is None or not c_el.text or c_el.text not in coordinates:
        continue
      fn_parent = t_scan.find("MultiplierFunction/function") if t_scan.find("MultiplierFunction") is not None else t_scan
      sc_scan = float(t_scan.find("MultiplierFunction/scale").text) if t_scan.find("MultiplierFunction") is not None else 1.0
      lf_el = fn_parent.find("LinearFunction/coefficients") if fn_parent is not None else None
      if lf_el is not None and lf_el.text:
        c_vals = sc_scan * str2vec(lf_el.text)
        if np.isclose(abs(c_vals[0]), 1.0) and np.isclose(c_vals[1], 0.0):
          coords_with_unit_axis.add(c_el.text)

    # Go through all six transforms
    transforms = ["rotation1", "rotation2", "rotation3", "translation1", "translation2", "translation3"]
    order = [3, 4, 5, 0, 1, 2]
    dof_designated = []
    for idx in order:

      t = transform_axes[idx]
      if t.attrib["name"] != transforms[idx]:
        logger.critical("Joints are parsed, or defined in the xml, in incorrect order")
        raise IndexError

      # Use the Coordinate parameters we parsed earlier; note that these do not exist for all joints (e.g
      # constant joints)
      coordinate = t.find("coordinates")
      if coordinate is not None and coordinate.text is not None and coordinate.text in coordinates:
        params = deepcopy(coordinates[coordinate.text])
      else:
        params = {"name": "{}_{}".format(xml.attrib["name"], t.attrib["name"]), "limited": False}

      params["name"] = f"{xml.attrib['name']}_{t.attrib['name']}"
      params["_transform_type"] = t.attrib["name"][:-1]

      # Set default reference position/angle to zero. If this value is not zero, then you need
      # more care while calculating quartic functions for equality constraints. This is not the same as
      # "default_value" for OpenSim Coordinate.
      params["ref"] = 0

      # Get axis of this transform
      axis = str2vec(t.find("axis").text)
      params["axis"] = estimate_axis(socket_child_frame, axis)

      # Figure out whether this is rotation or translation
      if params["_transform_type"] == 'rotation':
        params["type"] = "hinge"
      elif params["_transform_type"] == 'translation':
        params["type"] = "slide"
      else:
        logger.critical(f"Unidentified transformation type {params['_transform_type']} for transformation "
                        f"{params['name']}")
        raise TypeError

      # Parse the outermost MultiplierFunction if there is one
      scale = 1.0
      if t.find("MultiplierFunction") is not None:
        scale = float(t.find("MultiplierFunction/scale").text)
        t = t.find("MultiplierFunction/function")

      # See the comment before this loop. We have to designate one DoF per Coordinate owned by THIS joint
      # as an independent variable, i.e. make its dependence linear.
      if (
        coordinate is not None
        and coordinate.text is not None
        and coordinate.text in coordinates
        and coordinate.text not in dof_designated
      ):
        require_unit = coordinate.text in coords_with_unit_axis
        if self._designate_dof(t, params, coordinate, scale=scale, require_unit_linear=require_unit):
          dof_designated.append(coordinate.text)
          scale = 1.0

      # Handle a "Constant" transformation. We're not gonna create this joint
      # but we need the transformation information to properly align the joint
      flip_axis = False
      if t.find("Constant") is not None:

        # Get the value
        value = scale*float(t.find("Constant/value").text)

        # If the value is near zero don't bother creating this joint
        if abs(value) < 1e-6:
          continue

        # Otherwise define a locked joint
        params["limited"] = False
        params["_locked"] = True
        params["_transform_value"] = value

      # Handle a "SimmSpline", "NaturalCubicSpline", or "PiecewiseLinearFunction" transformation with a quartic approximation
      elif (
        t.find("SimmSpline") is not None
        or t.find("NaturalCubicSpline") is not None
        or t.find("PiecewiseLinearFunction") is not None
      ):

        # We can't model the relationship between two joints using a spline, but we can try to approximate it
        # with a quartic function. So fit a quartic function and check that the error is small enough

        # Get spline values
        if t.find("SimmSpline") is not None:
          x_values = t.find("SimmSpline/x")
          y_values = t.find("SimmSpline/y")
        elif t.find("NaturalCubicSpline") is not None:
          x_values = t.find("NaturalCubicSpline/x")
          y_values = t.find("NaturalCubicSpline/y")
        elif t.find("PiecewiseLinearFunction") is not None:
          x_values = t.find("PiecewiseLinearFunction/x")
          y_values = t.find("PiecewiseLinearFunction/y")
        else:
          logger.critical("Could not find spline definition in the XML element; was expecting to find it under "
                          "SimmSpline, NaturalCubicSpline, or PiecewiseLinearFunction")
          raise RuntimeError

        # Convert into numpy arrays; apply scaling to y values
        x_values = str2vec(x_values.text)
        y_values = scale*str2vec(y_values.text)

        fit, polycoef, joint_range = fit_spline(x_values, y_values)

        # Get the name of the independent coordinate
        independent_coordinate = coordinate.text

        # Create an output dir for plots (if it doesn't exist)
        output_dir = os.path.join(cfg.OUTPUT_PLOT_FOLDER, "custom_joints")
        if not os.path.isdir(output_dir):
          os.makedirs(output_dir)
        plot_and_save_figure(x_values, y_values, fit, params, independent_coordinate, output_dir)

        # Set range
        params["range"] = joint_range

        # Perhaps better set this joint as limited, since the behaviour outside of range can be bad
        params["limited"] = True

        # Update default value (if it is set; remember, it is set as "user")
        if "user" in params:
          params["user"] = fit(params["user"])

        # Update also _transform_value
        if "_transform_value" in params:
          params["_transform_value"] = fit(params["_transform_value"])

        # Let's check if the independent coordinate is dependent on another coordinate
        coordinate_mapping = cfg.OPENSIM.xpath(f"ConstraintSet//dependent_coordinate_name[text()='{coordinate.text}']")
        if len(coordinate_mapping) > 1:
          logger.critical(f"Coordinate {coordinate.text} depends on two other coordinates! This type of relation has "
                          f"not been implemented")
          raise NotImplementedError

        elif len(coordinate_mapping) == 1:

          # This coordinate depends on another coordinate
          constraint = coordinate_mapping[0].getparent()

          # Check if this dependency is enforced
          enforced = constraint.find("isEnforced")
          if enforced is not None and not str2bool(enforced.text):
            # This dependency is not enforced
            pass

          else:
            # Make sure this coordinate (again) depends only on one coordinate. Not sure how to check this, since
            # I've never seen a coordinate being dependent on multiple coordinates. Are they separated by commas,
            # or spaces?
            independent_coordinate = constraint.find("independent_coordinate_names").text
            if len(independent_coordinate.split(",")) > 1 or len(independent_coordinate.split(" ")) > 1:
              logger.critical(f"Coordinate {coordinate.text} depends on two other coordinates! This type of relation "
                              f"has not been implemented")
              raise NotImplementedError

            # Make sure the relation between these two coordinates is an identity mapping
            function = constraint.find("coupled_coordinates_function/LinearFunction")
            if function is None or not np.array_equal(str2vec(function.find("coefficients").text), np.array([1, 0])):
              logger.critical("Only linear mappings between two coordinates have been implemented.")
              raise NotImplementedError

        # Add a joint constraint between this joint and the independent joint, which we assume to be named
        # t["coordinate"]
        etree.SubElement(cfg.M_EQUALITY, "joint", joint1=params["name"], joint2=independent_coordinate,
                         polycoef=vec2str(polycoef),
                         solimp="0.9999 0.9999 0.001 0.5 2", active="true")

        # Update motion type to dependent for posterity
        params["_motion_type"] = "dependent"

      elif t.find("LinearFunction") is not None:

        coefficients = scale * str2vec(t.find("LinearFunction/coefficients").text)
        if np.isclose(abs(coefficients[0]), 1.0) and np.isclose(coefficients[1], 0.0):
          if params["name"] == (coordinate.text if coordinate is not None else None):
            # Designated independent coordinate DoF
            if coefficients[0] < 0:
              flip_axis = True
          else:
            # Secondary axis coupled with unit slope (+/-1) to independent coordinate
            independent_coordinate = coordinate.text
            polycoef = np.array([0.0, float(coefficients[0]), 0.0, 0.0, 0.0])
            etree.SubElement(
              cfg.M_EQUALITY,
              "joint",
              joint1=params["name"],
              joint2=independent_coordinate,
              polycoef=vec2str(polycoef),
              solimp="0.9999 0.9999 0.001 0.5 2",
              active="true",
            )
            params["_motion_type"] = "dependent"
        else:
          # Coupled linear function y = intercept + slope * q (e.g. cervical/lumbar spine or scapular rhythm)
          if coordinate is None or not coordinate.text:
            logger.critical(f"LinearFunction without coordinate in {xml.attrib['name']}/{t.attrib['name']}")
            raise RuntimeError
          independent_coordinate = coordinate.text
          slope, intercept = float(coefficients[0]), float(coefficients[1])
          polycoef = np.array([intercept, slope, 0.0, 0.0, 0.0])
          if "range" in params:
            mapped = np.sort(slope * np.asarray(params["range"], dtype=float) + intercept)
            if np.isclose(mapped[0], mapped[1]):
              mapped[0] -= 1e-6
            params["range"] = mapped
            params["limited"] = True
          if "user" in params:
            params["user"] = slope * float(params["user"]) + intercept
          if "_transform_value" in params:
            params["_transform_value"] = slope * float(params["_transform_value"]) + intercept
          etree.SubElement(
            cfg.M_EQUALITY,
            "joint",
            joint1=params["name"],
            joint2=independent_coordinate,
            polycoef=vec2str(polycoef),
            solimp="0.9999 0.9999 0.001 0.5 2",
            active="true",
          )
          params["_motion_type"] = "dependent"

      # Other functions are not defined yet
      else:
        logger.critical(f"Skipping transformation {t.attrib['name']} in joint {xml.attrib['name']}")
        raise RuntimeError

      # Calculate new axis
      if flip_axis:
        params["axis"] *= -1
        # Update default value too
        if "user" in params:
          params["user"] *= -1
        # And update _transform_value
        if "_transform_value" in params:
          params["_transform_value"] *= -1

      # Add joint
      joint = etree.SubElement(pointer, "joint", attrib=val2str(filter_keys(params)))

      # Add joint to list of joints
      all_joints.append(joint)

      # Add params to list of params
      all_params.append(params)

    # Check if all coordinates were designated as dofs. If there's a coordinate that wasn't designated, there's a good
    # chance that there exists an unnecessary equality constraint that describes the dependency of that coordinate to
    # another coordinate
    for coordinate_name in coordinates.keys():
      if coordinate_name not in dof_designated:
        c = cfg.M_EQUALITY.find(f"joint[@joint1='{coordinate_name}']")
        if c is not None:
          cfg.M_EQUALITY.remove(c)

    return all_joints, all_params

  def _designate_dof(self, t, params, coordinate, scale=1.0, require_unit_linear=False):
    """ Designate an OpenSim Coordinate as a MuJoCo DoF.

    Designate a Coordinate as a MuJoCo DoF (if applicable). This means that one of the MuJoCo joints (DoF) will
    correspond to an OpenSim Coordinate, although in OpenSim one coordinate can parameterize multiple DoFs. We only
    allow identity or linear mappings for designated DoFs.

    :param t: OpenSim sub-joint XML element
    :param params: Joint parameters
    :param coordinate: Copy of an OpenSim `Coordinate`
    :param scale: Optional MultiplierFunction scale factor
    :param require_unit_linear: If True, only designate an axis with unit linear slope ([+/-1, 0])
    :return: Boolean indicating whether a Coordinate has been designated.
    """

    # Find out the type of dependency
    spline_tag = None
    for candidate_tag in ("SimmSpline", "NaturalCubicSpline", "PiecewiseLinearFunction"):
      if t.find(candidate_tag) is not None:
        spline_tag = candidate_tag
        break

    if spline_tag is not None:

      # Check if the designated dof is a linear mapping -- if so, we should use it
      x_values = str2vec(t.find(f"{spline_tag}/x").text)
      y_values = scale * str2vec(t.find(f"{spline_tag}/y").text)
      fit, polycoef, joint_range = fit_spline(x_values, y_values)

      if len(x_values) > 2 and not np.all(np.isclose(x_values, y_values)):
        # Designated dof cannot (?) have a non-linear mapping
        return False
      if require_unit_linear and not (np.isclose(abs(polycoef[1]), 1.0) and np.isclose(polycoef[0], 0.0)):
        return False

      # Update range as min/max of the approximated range
      if "range" in params:
        params["range"] = np.sort(fit(params["range"]))
      else:
        params["range"] = joint_range

      # Update default value (if it is set; remember, it is set as "user")
      if "user" in params:
        params["user"] = fit(params["user"])

      # Update _transform_value too
      if "_transform_value" in params:
        params["_transform_value"] = fit(params["_transform_value"])

      # Make this into an identity mapping
      coefs = etree.SubElement(etree.SubElement(t, "LinearFunction"), "coefficients")
      coefs.text = "1 0"
      t.remove(t.find(spline_tag))

    elif t.find("LinearFunction") is not None:

      coefficients = scale * str2vec(t.find("LinearFunction/coefficients").text)
      slope, intercept = float(coefficients[0]), float(coefficients[1])
      if np.isclose(slope, 0.0):
        return False
      if require_unit_linear and not (np.isclose(abs(slope), 1.0) and np.isclose(intercept, 0.0)):
        return False

      polycoef = np.array([intercept, slope, 0.0, 0.0, 0.0])
      if not (np.isclose(abs(slope), 1.0) and np.isclose(intercept, 0.0)):
        abs_slope = abs(slope)
        signed_intercept = intercept if slope > 0 else -intercept
        if "range" in params:
          mapped = np.sort(abs_slope * np.asarray(params["range"], dtype=float) + signed_intercept)
          if np.isclose(mapped[0], mapped[1]):
            mapped[0] -= 1e-6
          params["range"] = mapped
        if "user" in params:
          params["user"] = abs_slope * float(params["user"]) + signed_intercept
        if "_transform_value" in params:
          params["_transform_value"] = abs_slope * float(params["_transform_value"]) + signed_intercept
        t.find("LinearFunction/coefficients").text = "1 0" if slope > 0 else "-1 0"

    elif t.find("Constant") is not None:
      return False

    else:
      logger.critical("Only LinearFunction and spline transformation types are currently implemented")
      raise NotImplementedError

    # Range of designated dof may have been modified above, if the mapping between Coordinate and designated dof is
    # not identity. In that case, we need to update the dependencies (polycoefs) of constraints that depend on this
    # Coordinate/dof
    for constraint in cfg.M_EQUALITY.findall(f"joint[@joint2='{coordinate.text}']"):
      constraint_polycoef = str2vec(constraint.attrib["polycoef"])
      if np.allclose(polycoef, np.array([0, 1, 0, 0, 0])) or np.allclose(polycoef, np.array([0, -1, 0, 0, 0])):
        if polycoef[1] < 0:
          # Axis will be flipped below, so q_mjc == q_osim
          pass
      else:
        # q_mjc = |a| * q_osim + signed_b  =>  q_osim = (q_mjc - signed_b) / |a|
        abs_a = abs(polycoef[1])
        signed_b = polycoef[0] if polycoef[1] > 0 else -polycoef[0]
        p_orig = np.polynomial.polynomial.Polynomial(constraint_polycoef)
        q_sub = np.polynomial.polynomial.Polynomial([-signed_b / abs_a, 1.0 / abs_a])
        composed = (p_orig(q_sub)).coef
        new_poly = np.zeros((5,))
        new_poly[: min(5, len(composed))] = composed[:5]
        constraint.attrib["polycoef"] = vec2str(new_poly)

    # Designate this dof as 'coordinate.text'
    params["name"] = coordinate.text

    return True