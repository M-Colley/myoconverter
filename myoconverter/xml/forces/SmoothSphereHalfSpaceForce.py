""" Contains the `SmoothSphereHalfSpaceForce` parser.

Converts OpenSim 4.x `SmoothSphereHalfSpaceForce` elements (used in clinical
gait, Moco, cerebral palsy, stroke, and prosthetic `.osim` models) into
MJX-compatible MuJoCo sphere-plane `<geom>` + `<contact><pair>` definitions.
"""

from lxml import etree

from myoconverter.xml import config as cfg
from myoconverter.xml.forces.utils import ensure_contact_geometry
from myoconverter.xml.parsers import IParser
from myoconverter.xml.utils import element_txt2num, str2bool


class SmoothSphereHalfSpaceForce(IParser):
  """ This class parses and converts the OpenSim `SmoothSphereHalfSpaceForce` XML element to MuJoCo. """

  def parse(self, xml):
    if xml.find("appliesForce") is not None and not str2bool(xml.find("appliesForce").text):
      return

    sphere_ref = None
    for tag in ("socket_sphere", "sphere"):
      el = xml.find(tag)
      if el is not None and el.text is not None and el.text.strip():
        sphere_ref = el.text.strip()
        break

    hs_ref = None
    for tag in ("socket_half_space", "half_space"):
      el = xml.find(tag)
      if el is not None and el.text is not None and el.text.strip():
        hs_ref = el.text.strip()
        break

    g_sphere = ensure_contact_geometry(sphere_ref)
    g_plane = ensure_contact_geometry(hs_ref)
    if not g_sphere or not g_plane:
      return

    mu_s = element_txt2num(xml, "static_friction", default=0.8)
    mu_d = element_txt2num(xml, "dynamic_friction", default=mu_s)
    mu = float(max(0.01, 0.5 * (mu_s + mu_d))) if (mu_s == mu_s and mu_d == mu_d) else 0.8

    pair_name = xml.attrib.get("name", f"{g_sphere}_{g_plane}_pair")
    if cfg.M_CONTACT.find(f"pair[@geom1='{g_sphere}'][@geom2='{g_plane}']") is None:
      etree.SubElement(
        cfg.M_CONTACT,
        "pair",
        name=pair_name,
        geom1=g_sphere,
        geom2=g_plane,
        friction=f"{mu:.4g} {mu:.4g} 0.005 0.0001 0.0001",
      )
