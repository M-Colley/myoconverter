""" Contains the `HuntCrossleyForce` parser.

Converts OpenSim 4.x `HuntCrossleyForce` contact sets (used across SimTK gait,
prosthetic, clinical, and animal `.osim` models) into MJX-compatible MuJoCo
`<geom>` + `<contact><pair>` definitions.
"""

from lxml import etree

from myoconverter.xml import config as cfg
from myoconverter.xml.forces.utils import ensure_contact_geometry
from myoconverter.xml.parsers import IParser
from myoconverter.xml.utils import element_txt2num, str2bool


class HuntCrossleyForce(IParser):
  """ This class parses and converts the OpenSim `HuntCrossleyForce` XML element to MuJoCo. """

  def parse(self, xml):
    if xml.find("appliesForce") is not None and not str2bool(xml.find("appliesForce").text):
      return

    geom_names = []
    mu = 0.8
    for geom_el in xml.findall(".//geometry"):
      if geom_el.text:
        geom_names.extend([tok.strip() for tok in geom_el.text.split() if tok.strip()])
      parent = geom_el.getparent()
      if parent is not None:
        mu_s = element_txt2num(parent, "static_friction", default=0.8)
        mu_d = element_txt2num(parent, "dynamic_friction", default=mu_s)
        if mu_s == mu_s and mu_d == mu_d:
          mu = float(max(0.01, 0.5 * (mu_s + mu_d)))

    materialized = []
    for gref in geom_names:
      mname = ensure_contact_geometry(gref)
      if mname and mname not in materialized:
        materialized.append(mname)

    planes = []
    others = []
    for mname in materialized:
      g_el = cfg.M_WORLDBODY.find(f".//geom[@name='{mname}']")
      if g_el is not None and g_el.attrib.get("type") == "plane":
        planes.append(mname)
      elif g_el is not None:
        others.append(mname)

    if planes and others:
      for g_other in others:
        for g_plane in planes:
          if cfg.M_CONTACT.find(f"pair[@geom1='{g_other}'][@geom2='{g_plane}']") is None:
            etree.SubElement(
              cfg.M_CONTACT,
              "pair",
              geom1=g_other,
              geom2=g_plane,
              friction=f"{mu:.4g} {mu:.4g} 0.005 0.0001 0.0001",
            )
    elif len(others) >= 2:
      for i in range(len(others)):
        for j in range(i + 1, len(others)):
          g1, g2 = others[i], others[j]
          if cfg.M_CONTACT.find(f"pair[@geom1='{g1}'][@geom2='{g2}']") is None:
            etree.SubElement(
              cfg.M_CONTACT,
              "pair",
              geom1=g1,
              geom2=g2,
              friction=f"{mu:.4g} {mu:.4g} 0.005 0.0001 0.0001",
            )
