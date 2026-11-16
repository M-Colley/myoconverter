"""End-to-end tests for SimTK clinical, surgical, prosthetic, spine, and non-human `.osim` parsers."""

import json
import mujoco
import numpy as np
import pytest

from myoconverter.O2MPipeline import O2MPipeline
from myoconverter.xml.converter import convert


def _osim_doc(
    bodies_xml: str,
    joints_xml: str,
    forces_xml: str = "",
    constraints_xml: str = "",
    contact_geoms_xml: str = "",
) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
  <Model name="simtk_clinical_test">
    <Ground name="ground">
      <attached_geometry />
      <WrapObjectSet name="wrapobjectset"><objects /></WrapObjectSet>
    </Ground>
    <BodySet name="bodyset">
      <objects>
        {bodies_xml}
      </objects>
    </BodySet>
    <JointSet name="jointset">
      <objects>
        {joints_xml}
      </objects>
    </JointSet>
    <ConstraintSet name="constraintset">
      <objects>
        {constraints_xml}
      </objects>
    </ConstraintSet>
    <ForceSet name="forceset">
      <objects>
        {forces_xml}
      </objects>
    </ForceSet>
    <MarkerSet name="markerset"><objects /></MarkerSet>
    <ContactGeometrySet name="contactgeometryset">
      <objects>
        {contact_geoms_xml}
      </objects>
    </ContactGeometrySet>
  </Model>
</OpenSimDocument>
"""


def _body(name: str, mass: float = 1.0) -> str:
    return f"""
    <Body name="{name}">
      <mass>{mass}</mass>
      <mass_center>0 0 0</mass_center>
      <inertia>0.01 0.01 0.01 0 0 0</inertia>
      <attached_geometry />
      <WrapObjectSet name="wrapobjectset"><objects /></WrapObjectSet>
    </Body>
    """


def _coord(name: str, rmin: float = -1.5, rmax: float = 1.5, default: float = 0.0) -> str:
    return f"""
    <Coordinate name="{name}">
      <default_value>{default}</default_value>
      <range>{rmin} {rmax}</range>
      <clamped>true</clamped>
      <locked>false</locked>
    </Coordinate>
    """


def _frames(jname: str, parent: str, child: str, p_trans: str = "0 0 0") -> str:
    p_sock = "/ground" if parent == "ground" else f"/bodyset/{parent}"
    return f"""
    <socket_parent_frame>{jname}_p</socket_parent_frame>
    <socket_child_frame>{jname}_c</socket_child_frame>
    <frames>
      <PhysicalOffsetFrame name="{jname}_p">
        <socket_parent>{p_sock}</socket_parent>
        <translation>{p_trans}</translation>
        <orientation>0 0 0</orientation>
      </PhysicalOffsetFrame>
      <PhysicalOffsetFrame name="{jname}_c">
        <socket_parent>/bodyset/{child}</socket_parent>
        <translation>0 0 0</translation>
        <orientation>0 0 0</orientation>
      </PhysicalOffsetFrame>
    </frames>
    """


def test_free_ball_gimbal_planar_and_ellipsoid_joints_compile(tmp_path):
    bodies = (
        _body("pelvis", 10.0)
        + _body("femur", 5.0)
        + _body("spine_seg", 4.0)
        + _body("scapula", 1.5)
        + _body("foot_2d", 1.0)
    )
    joints = f"""
    <FreeJoint name="ground_pelvis">
      {_frames("ground_pelvis", "ground", "pelvis", "0 0.9 0")}
      <coordinates>
        {_coord("pelvis_rx")}{_coord("pelvis_ry")}{_coord("pelvis_rz")}
        {_coord("pelvis_tx", -5, 5)}{_coord("pelvis_ty", -1, 3, 0.9)}{_coord("pelvis_tz", -5, 5)}
      </coordinates>
    </FreeJoint>
    <BallJoint name="hip_r">
      {_frames("hip_r", "pelvis", "femur", "0 -0.1 0.08")}
      <coordinates>
        {_coord("hip_flex")}{_coord("hip_add")}{_coord("hip_rot")}
      </coordinates>
    </BallJoint>
    <GimbalJoint name="lumbar">
      {_frames("lumbar", "pelvis", "spine_seg", "0 0.1 0")}
      <coordinates>
        {_coord("lumbar_ext")}{_coord("lumbar_bend")}{_coord("lumbar_rot")}
      </coordinates>
    </GimbalJoint>
    <EllipsoidJoint name="scapulothoracic">
      {_frames("scapulothoracic", "spine_seg", "scapula", "0 0.15 0.12")}
      <radii_x_y_z>0.08 0.15 0.09</radii_x_y_z>
      <coordinates>
        {_coord("scap_abd")}{_coord("scap_elev")}{_coord("scap_up_rot")}
      </coordinates>
    </EllipsoidJoint>
    <PlanarJoint name="ankle_planar">
      {_frames("ankle_planar", "femur", "foot_2d", "0 -0.4 0")}
      <coordinates>
        {_coord("foot_rz")}{_coord("foot_tx", -0.1, 0.1)}{_coord("foot_ty", -0.1, 0.1)}
      </coordinates>
    </PlanarJoint>
    """
    osim_path = tmp_path / "simtk_joints.osim"
    osim_path.write_text(_osim_doc(bodies, joints), encoding="utf-8")

    out_xml = convert(str(osim_path), str(tmp_path / "out"))
    model = mujoco.MjModel.from_xml_path(out_xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # 6 (Free) + 3 (Ball) + 3 (Gimbal) + 3 (Ellipsoid) + 3 (Planar) = 18 DoF
    assert model.nq == 18
    assert np.all(np.isfinite(data.qacc))


def test_custom_joint_fractional_linear_spine_rhythm_and_piecewise_coupler(tmp_path):
    bodies = _body("c7") + _body("c6")
    joints = f"""
    <CustomJoint name="c7_jnt">
      {_frames("c7_jnt", "ground", "c7", "0 0.5 0")}
      <coordinates>{_coord("neck_flex", -0.8, 0.8, 0.1)}</coordinates>
      <SpatialTransform>
        <TransformAxis name="rotation1"><coordinates></coordinates><axis>1 0 0</axis><Constant><value>0</value></Constant></TransformAxis>
        <TransformAxis name="rotation2"><coordinates></coordinates><axis>0 1 0</axis><Constant><value>0</value></Constant></TransformAxis>
        <TransformAxis name="rotation3"><coordinates>neck_flex</coordinates><axis>0 0 1</axis><LinearFunction><coefficients>1 0</coefficients></LinearFunction></TransformAxis>
        <TransformAxis name="translation1"><coordinates>neck_flex</coordinates><axis>1 0 0</axis><LinearFunction><coefficients>0.015 0</coefficients></LinearFunction></TransformAxis>
        <TransformAxis name="translation2"><coordinates></coordinates><axis>0 1 0</axis><Constant><value>0</value></Constant></TransformAxis>
        <TransformAxis name="translation3"><coordinates></coordinates><axis>0 0 1</axis><Constant><value>0</value></Constant></TransformAxis>
      </SpatialTransform>
    </CustomJoint>
    <CustomJoint name="c6_jnt">
      {_frames("c6_jnt", "c7", "c6", "0 0.03 0")}
      <coordinates>{_coord("c6_flex", -0.5, 0.5, 0.0)}</coordinates>
      <SpatialTransform>
        <TransformAxis name="rotation1"><coordinates></coordinates><axis>1 0 0</axis><Constant><value>0</value></Constant></TransformAxis>
        <TransformAxis name="rotation2"><coordinates></coordinates><axis>0 1 0</axis><Constant><value>0</value></Constant></TransformAxis>
        <TransformAxis name="rotation3"><coordinates>c6_flex</coordinates><axis>0 0 1</axis><LinearFunction><coefficients>1 0</coefficients></LinearFunction></TransformAxis>
        <TransformAxis name="translation1"><coordinates></coordinates><axis>1 0 0</axis><Constant><value>0</value></Constant></TransformAxis>
        <TransformAxis name="translation2"><coordinates></coordinates><axis>0 1 0</axis><Constant><value>0</value></Constant></TransformAxis>
        <TransformAxis name="translation3"><coordinates></coordinates><axis>0 0 1</axis><Constant><value>0</value></Constant></TransformAxis>
      </SpatialTransform>
    </CustomJoint>
    """
    constraints = """
    <CoordinateCouplerConstraint name="cervical_rhythm">
      <isEnforced>true</isEnforced>
      <independent_coordinate_names>neck_flex</independent_coordinate_names>
      <dependent_coordinate_name>c6_flex</dependent_coordinate_name>
      <coupled_coordinates_function>
        <PiecewiseLinearFunction>
          <x>-0.8 0.0 0.8</x>
          <y>-0.28 0.0 0.28</y>
        </PiecewiseLinearFunction>
      </coupled_coordinates_function>
    </CoordinateCouplerConstraint>
    """
    osim_path = tmp_path / "cervical_rhythm.osim"
    osim_path.write_text(_osim_doc(bodies, joints, constraints_xml=constraints), encoding="utf-8")

    out_xml = convert(str(osim_path), str(tmp_path / "out"))
    model = mujoco.MjModel.from_xml_path(out_xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    jid_tx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "c7_jnt_translation1")
    jid_c6 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "c6_flex")
    assert jid_tx >= 0 and jid_c6 >= 0
    assert np.isclose(data.qpos[model.jnt_qposadr[jid_tx]], 0.015 * 0.1, atol=1e-4)
    assert np.isclose(data.qpos[model.jnt_qposadr[jid_c6]], 0.35 * 0.1, atol=1e-3)


def test_surgical_weld_point_and_constant_distance_constraints(tmp_path):
    bodies = _body("residual_tibia") + _body("prosthetic_socket") + _body("link_bar")
    joints = f"""
    <PinJoint name="knee">
      {_frames("knee", "ground", "residual_tibia", "0 0.5 0")}
      <coordinates>{_coord("knee_angle")}</coordinates>
    </PinJoint>
    <PinJoint name="socket_dof">
      {_frames("socket_dof", "residual_tibia", "prosthetic_socket", "0 -0.2 0")}
      <coordinates>{_coord("socket_flex")}</coordinates>
    </PinJoint>
    <SliderJoint name="bar_slide">
      {_frames("bar_slide", "prosthetic_socket", "link_bar", "0.1 0 0")}
      <coordinates>{_coord("bar_tx", -0.2, 0.2)}</coordinates>
    </SliderJoint>
    """
    constraints = """
    <WeldConstraint name="socket_fusion">
      <isEnforced>true</isEnforced>
      <socket_frame1>tibia_mount</socket_frame1>
      <socket_frame2>socket_mount</socket_frame2>
      <frames>
        <PhysicalOffsetFrame name="tibia_mount">
          <socket_parent>/bodyset/residual_tibia</socket_parent>
          <translation>0 -0.2 0</translation>
          <orientation>0 0 0</orientation>
        </PhysicalOffsetFrame>
        <PhysicalOffsetFrame name="socket_mount">
          <socket_parent>/bodyset/prosthetic_socket</socket_parent>
          <translation>0 0 0</translation>
          <orientation>0 0 0</orientation>
        </PhysicalOffsetFrame>
      </frames>
    </WeldConstraint>
    <PointConstraint name="fourbar_pin">
      <isEnforced>true</isEnforced>
      <socket_body_1>/bodyset/prosthetic_socket</socket_body_1>
      <socket_body_2>/bodyset/link_bar</socket_body_2>
      <location_body_1>0.1 0 0</location_body_1>
      <location_body_2>0 0 0</location_body_2>
    </PointConstraint>
    <ConstantDistanceConstraint name="fixator_rod">
      <isEnforced>true</isEnforced>
      <socket_body_1>/bodyset/residual_tibia</socket_body_1>
      <socket_body_2>/bodyset/prosthetic_socket</socket_body_2>
      <location_body_1>0 0 0</location_body_1>
      <location_body_2>0 0 0</location_body_2>
      <constant_distance>0.2</constant_distance>
    </ConstantDistanceConstraint>
    """
    osim_path = tmp_path / "prosthetic_constraints.osim"
    osim_path.write_text(_osim_doc(bodies, joints, constraints_xml=constraints), encoding="utf-8")

    out_xml = convert(str(osim_path), str(tmp_path / "out"))
    model = mujoco.MjModel.from_xml_path(out_xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    assert model.neq == 3
    assert np.allclose(data.efc_pos, 0.0, atol=1e-5)


def test_simtk_muscles_actuators_expression_force_and_contacts(tmp_path):
    bodies = _body("shank") + _body("foot")
    joints = f"""
    <PinJoint name="ankle">
      {_frames("ankle", "ground", "shank", "0 0.4 0")}
      <coordinates>{_coord("ankle_angle", -0.6, 0.6, 0.0)}</coordinates>
    </PinJoint>
    <PinJoint name="mtp">
      {_frames("mtp", "shank", "foot", "0 -0.35 0")}
      <coordinates>{_coord("mtp_angle", -0.5, 0.5, 0.0)}</coordinates>
    </PinJoint>
    """
    contact_geoms = """
    <ContactHalfSpace name="floor">
      <socket_frame>/ground</socket_frame>
      <location>0 0 0</location>
      <orientation>0 0 -1.5707963267948966</orientation>
    </ContactHalfSpace>
    <ContactSphere name="heel_sphere">
      <socket_frame>/bodyset/foot</socket_frame>
      <location>-0.03 -0.02 0</location>
      <orientation>0 0 0</orientation>
      <radius>0.03</radius>
    </ContactSphere>
    <ContactSphere name="toe_sphere">
      <socket_frame>/bodyset/foot</socket_frame>
      <location>0.10 -0.02 0</location>
      <orientation>0 0 0</orientation>
      <radius>0.025</radius>
    </ContactSphere>
    """

    def _muscle(tag: str, name: str) -> str:
        return f"""
        <{tag} name="{name}">
          <max_isometric_force>600</max_isometric_force>
          <optimal_fiber_length>0.08</optimal_fiber_length>
          <tendon_slack_length>0.22</tendon_slack_length>
          <pennation_angle_at_optimal>0.1</pennation_angle_at_optimal>
          <GeometryPath name="path">
            <PathPointSet><objects>
              <PathPoint name="{name}_p1"><socket_parent_frame>/bodyset/shank</socket_parent_frame><location>0 -0.1 0.02</location></PathPoint>
              <PathPoint name="{name}_p2"><socket_parent_frame>/bodyset/foot</socket_parent_frame><location>-0.02 0.01 0</location></PathPoint>
            </objects></PathPointSet>
          </GeometryPath>
        </{tag}>
        """

    forces = (
        _muscle("DeGrooteFregly2016Muscle", "soleus_dgf")
        + _muscle("Millard2012AccelerationMuscle", "tib_ant_acc")
        + _muscle("Delp1990Muscle_Deprecated", "gastroc_delp")
        + _muscle("RigidTendonMuscle", "peroneus_rt")
        + """
        <ActivationCoordinateActuator name="exo_assist">
          <coordinate>ankle_angle</coordinate>
          <optimal_force>45.0</optimal_force>
          <activation_time_constant>0.02</activation_time_constant>
          <min_control>-1</min_control>
          <max_control>1</max_control>
        </ActivationCoordinateActuator>
        <ExpressionBasedCoordinateForce name="mtp_spring">
          <coordinate>mtp_angle</coordinate>
          <expression>-12.0*(q - 0.05) - 0.4*qdot</expression>
        </ExpressionBasedCoordinateForce>
        <SmoothSphereHalfSpaceForce name="heel_contact">
          <socket_sphere>heel_sphere</socket_sphere>
          <socket_half_space>floor</socket_half_space>
          <static_friction>0.85</static_friction>
        </SmoothSphereHalfSpaceForce>
        <HuntCrossleyForce name="toe_contact">
          <contact_parameters>
            <ContactParameters>
              <geometry>toe_sphere floor</geometry>
              <static_friction>0.8</static_friction>
              <dynamic_friction>0.75</dynamic_friction>
            </ContactParameters>
          </contact_parameters>
        </HuntCrossleyForce>
        """
    )

    osim_path = tmp_path / "clinical_gait_exo.osim"
    osim_path.write_text(
        _osim_doc(bodies, joints, forces_xml=forces, contact_geoms_xml=contact_geoms),
        encoding="utf-8",
    )

    out_xml = convert(str(osim_path), str(tmp_path / "out"))
    model = mujoco.MjModel.from_xml_path(out_xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # 1 ActivationCoordinateActuator + 4 muscles = 5 actuators, 2 sphere-plane contact pairs
    assert model.nu == 5
    assert model.npair == 2

    mtp_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mtp_angle")
    assert np.isclose(model.jnt_stiffness[mtp_jid], 12.0, atol=1e-3)
    assert np.isclose(model.qpos_spring[model.jnt_qposadr[mtp_jid]], 0.05, atol=1e-3)

    # Floor plane outward normal in MuJoCo worldbody must point +Z (up; Ground euler uses 4-sig-fig 1.571 rad)
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    world_z_axis = data.geom_xmat[floor_gid].reshape(3, 3)[:, 2]
    assert np.allclose(world_z_axis, [0.0, 0.0, 1.0], atol=1e-3)


def test_pure_python_pipeline_lenient_scaleset_mjx_and_model_card(tmp_path):
    geom_dir = tmp_path / "Geometry"
    geom_dir.mkdir()
    bodies = _body("arm_seg", 2.0)
    joints = f"""
    <PinJoint name="elbow">
      {_frames("elbow", "ground", "arm_seg", "0 0.5 0")}
      <coordinates>{_coord("elbow_flex", 0.0, 2.2, 0.5)}</coordinates>
    </PinJoint>
    """
    forces = """
    <DeGrooteFregly2016Muscle name="biceps_dgf">
      <max_isometric_force>400</max_isometric_force>
      <optimal_fiber_length>0.12</optimal_fiber_length>
      <tendon_slack_length>0.18</tendon_slack_length>
      <pennation_angle_at_optimal>0.0</pennation_angle_at_optimal>
      <GeometryPath name="path">
        <PathPointSet><objects>
          <PathPoint name="p1"><socket_parent_frame>/ground</socket_parent_frame><location>0 0.65 0.03</location></PathPoint>
          <PathPoint name="p2"><socket_parent_frame>/bodyset/arm_seg</socket_parent_frame><location>0 -0.08 0.03</location></PathPoint>
        </objects></PathPointSet>
      </GeometryPath>
    </DeGrooteFregly2016Muscle>
    <CustomSimTKPluginForce name="exotic_plugin_probe">
      <some_param>42</some_param>
    </CustomSimTKPluginForce>
    """
    osim_path = tmp_path / "patient_arm.osim"
    osim_path.write_text(_osim_doc(bodies, joints, forces_xml=forces), encoding="utf-8")

    scaleset_path = tmp_path / "patient_scales.xml"
    scaleset_path.write_text(
        """<?xml version="1.0" encoding="UTF-8" ?>
        <OpenSimDocument Version="40000">
          <ScaleSet name="patient_scales">
            <objects>
              <Scale>
                <scales>1.08 1.08 1.08</scales>
                <segment>arm_seg</segment>
                <apply>true</apply>
              </Scale>
            </objects>
          </ScaleSet>
        </OpenSimDocument>
        """,
        encoding="utf-8",
    )

    # Strict mode rejects unknown custom plugin tags
    with pytest.raises(NotImplementedError):
        O2MPipeline(
            str(osim_path),
            str(geom_dir),
            str(tmp_path / "out_strict"),
            convert_steps=[1],
            validation=False,
        )

    # Lenient mode + ScaleSet + MJX + Model Card succeeds end-to-end in pure Python!
    out_dir = tmp_path / "out_fork"
    O2MPipeline(
        str(osim_path),
        str(geom_dir),
        str(out_dir),
        convert_steps=[1],
        validation=False,
        lenient_parsers=True,
        scale_set=str(scaleset_path),
        mjx=True,
        export_model_card=True,
    )

    assert (out_dir / "patient_arm_cvt1.xml").is_file()
    assert (out_dir / "patient_arm_cvt1_scaled.xml").is_file()
    mjx_xml = out_dir / "patient_arm_cvt1_scaled_mjx.xml"
    assert mjx_xml.is_file()
    assert (out_dir / "patient_arm_mjx_report.json").is_file()

    card_path = out_dir / "patient_arm_model_card.json"
    assert card_path.is_file()
    card = json.loads(card_path.read_text(encoding="utf-8"))
    assert card["skipped_opensim_elements"] == [
        {"component": "forces", "tag": "CustomSimTKPluginForce", "name": "exotic_plugin_probe"}
    ]

    compiled = mujoco.MjModel.from_xml_path(str(mjx_xml))
    data = mujoco.MjData(compiled)
    mujoco.mj_step(compiled, data)
    assert np.all(np.isfinite(data.qacc))
