#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Thu May  5 11:16:03 2022
# @author: Huawei Wang

# # PyOpenGL/MuJoCo fails in Linux if running with something else than osmesa; needs to be set before mujoco is imported
# import platform
# if platform.system() == "Linux":
#     import os
#     os.environ["MUJOCO_GL"] = "osmesa"

import argparse
from loguru import logger
from myoconverter.conversion_steps.O2MSteps import O2MSteps
import pickle
import os

def str2bool(value):
    """Parse flexible CLI boolean values without bool('False') == True pitfalls."""
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y", "t"}:
        return True
    if lowered in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def parse_convert_steps(values):
    """Normalize CLI --convert_steps tokens (e.g. `1 2 3`, `1,2`, or `[1, 2]`) into a list of ints."""
    if isinstance(values, int):
        values = [values]
    steps = []
    for item in values:
        if isinstance(item, int):
            tokens = [str(item)]
        else:
            cleaned = str(item).strip().strip("[]()")
            tokens = [tok.strip() for tok in cleaned.replace(",", " ").split() if tok.strip()]
        for tok in tokens:
            try:
                step = int(tok)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"Invalid conversion step {tok!r}; expected 1, 2, or 3") from exc
            if step not in {1, 2, 3}:
                raise argparse.ArgumentTypeError(f"Conversion step must be 1, 2, or 3, got {step}")
            if step not in steps:
                steps.append(step)
    if not steps:
        raise argparse.ArgumentTypeError("At least one conversion step in {1, 2, 3} must be provided")
    return steps


def parse_muscle_list(values):
    """Normalize CLI --muscle_list tokens (space- or comma-separated) into a list of muscle names."""
    if values is None:
        return None
    muscles = []
    for item in values:
        for tok in str(item).split(","):
            name = tok.strip()
            if name:
                muscles.append(name)
    return muscles or None


def O2MPipeline(osim_file, geometry_folder, output_folder, **kwargs):
    """
    :param osim_file: Path to the OpenSim OSIM model file
    :param geometry_folder: Path to the Geometry folder
    :param output_folder: Path to folder where converted model is saved
    :param kwargs: convert_steps, muscle_list, osim_data_overwrite, conversion,
        validation, generate_pdf, speedy, add_ground_geom,
        treat_as_normal_path_point
    :return:
    """

    # based on updated configure options, run the pipeline
    convert_steps = kwargs.get("convert_steps", [1, 2, 3])
    muscle_list = kwargs.get("muscle_list", None)
    osim_data_overwrite = kwargs.get("osim_data_overwrite", False)
    conversion = kwargs.get("conversion", True)
    validation = kwargs.get("validation", True)
    generate_pdf = kwargs.get("generate_pdf", False)
    speedy = kwargs.get("speedy", False)
    add_ground_geom = kwargs.get("add_ground_geom", False)
    treat_as_normal_path_point = kwargs.get("treat_as_normal_path_point", False)
    lenient_parsers = kwargs.get("lenient_parsers", False)
    strict_parsers = kwargs.get("strict_parsers", not lenient_parsers)
    mjx = kwargs.get("mjx", False)
    mjx_integrator = kwargs.get("mjx_integrator", "implicitfast")
    scale_set = kwargs.get("scale_set", None)
    export_card = kwargs.get("export_model_card", False)

    # set logging
    os.makedirs(output_folder, exist_ok=True)
    MODEL_NAME = os.path.splitext(os.path.basename(osim_file))[0]
    OUTPUT_LOG_FILE = os.path.join(output_folder, f"{MODEL_NAME}_conversion.log")

    # If there is an existing log file, remove it
    if os.path.exists(OUTPUT_LOG_FILE):
        os.remove(OUTPUT_LOG_FILE)

    # Add scoped file handler and ensure only this handler is removed on exit
    log_id = logger.add(OUTPUT_LOG_FILE)
    try:
        logger.info(f"Start the conversion pipeline for : {MODEL_NAME}")

        # coordinate configure options
        osim_data_overwrite = coordinate_kwargs(
            osim_file,
            output_folder,
            convert_steps,
            osim_data_overwrite,
            conversion,
            validation,
            speedy,
        )

        O2MSteps_inst = O2MSteps(
            osim_file,
            geometry_folder,
            output_folder,
            convert_steps=convert_steps,
            muscle_list=muscle_list,
            osim_data_overwrite=osim_data_overwrite,
            convert=conversion,
            validation=validation,
            generate_pdf=generate_pdf,
            speedy=speedy,
            add_ground_geom=add_ground_geom,
            treat_as_normal_path_point=treat_as_normal_path_point,
            strict_parsers=strict_parsers,
        )

        O2MSteps_inst.PipelineExecution()

        # Discover latest produced MJCF file for optional post-processing steps
        latest_mjcf = None
        for step_num in (3, 2, 1):
            candidate = os.path.join(output_folder, f"{MODEL_NAME}_cvt{step_num}.xml")
            if os.path.isfile(candidate):
                latest_mjcf = candidate
                break

        if scale_set and latest_mjcf:
            from myoconverter.utils.mjcf_scaling import scale_converted_mjcf

            logger.info(f"Applying OpenSim ScaleSet '{scale_set}' to {latest_mjcf}")
            latest_mjcf = str(scale_converted_mjcf(latest_mjcf, scale_set_xml=scale_set))

        if mjx and latest_mjcf:
            import json
            from myoconverter.utils.mjx_postprocess import make_mjx_ready

            logger.info(f"Post-processing {latest_mjcf} for MJX (integrator={mjx_integrator})")
            mjx_report = make_mjx_ready(latest_mjcf, integrator=mjx_integrator)
            latest_mjcf = mjx_report.get("output_xml", latest_mjcf)
            report_path = os.path.join(output_folder, f"{MODEL_NAME}_mjx_report.json")
            with open(report_path, "w", encoding="utf-8") as fp:
                json.dump(mjx_report, fp, indent=2)

        if export_card and latest_mjcf:
            import json
            from myoconverter.utils.model_card import export_model_card
            from myoconverter.xml import config as xml_cfg

            card_path = export_model_card(
                output_folder,
                MODEL_NAME,
                osim_file=osim_file,
                mjcf_file=latest_mjcf,
                convert_steps=convert_steps,
            )
            skipped = getattr(xml_cfg, "SKIPPED_ELEMENTS", None)
            if skipped:
                with open(card_path, "r", encoding="utf-8") as fp:
                    card_data = json.load(fp)
                card_data["skipped_opensim_elements"] = list(skipped)
                with open(card_path, "w", encoding="utf-8") as fp:
                    json.dump(card_data, fp, indent=2)
    finally:
        logger.remove(log_id)

def coordinate_kwargs(osim_file, output_folder, convert_steps,\
                                   osim_data_overwrite, conversion, validation, speedy):
    """
    Some config flags may conflicting with each other, this coordinate step is to manage them.
    """

    # write down the configure coordinate info
    logger.info("Coordinate the input configurations, remove conflicts.")

    # if conversion is not selected, then no need to regenerate osim dataset
    # Give back this control to users ...
    # if not conversion:
    #     osim_data_overwrite = False
    #     logger.info("   Optimization process was not selected, set osim_data_overwrite to FALSE.")

    # then check the last saved configure files in the second and third steps
    if 2 in convert_steps:

        logger.info("   Checking configurations for step 2 conversion.")

        if os.path.isfile(output_folder + '/Step2_muscleKinematics/config.pkl'):
            with open(output_folder + '/Step2_muscleKinematics/config.pkl', 'rb') as old_configure_file:
                old_config = pickle.load(old_configure_file)

            # if speedy flag changes, overwriting is always needed for optimization
            if old_config["speedy"] != speedy:
                if conversion:
                    osim_data_overwrite = True

                    logger.info("       Speedy flag is different from last run, set osim_data_overwrite to TRUE.")

        else:
            osim_data_overwrite = True
            logger.info("       Configure of previous run does not exist, set osim_data_overwrite to TRUE.")

        logger.info("       All good now.")

        # save kwargs
        kwargs = {}
        kwargs["convert_steps"] = convert_steps
        kwargs["osim_data_overwrite"] = osim_data_overwrite
        kwargs["conversion"] = conversion
        kwargs["speedy"] = speedy

        # save the update kwargs to config.pkl in step 2 folder
        os.makedirs(output_folder + '/Step2_muscleKinematics', exist_ok = True)
        with open(output_folder + '/Step2_muscleKinematics/config.pkl', 'wb') as config_save:
            pickle.dump(kwargs, config_save)

    if 3 in convert_steps:

        logger.info("   Checking configurations for step 3 conversion.")

        if os.path.isfile(output_folder + '/Step3_muscleKinetics/config.pkl'):
            with open(output_folder + '/Step3_muscleKinetics/config.pkl', 'rb') as old_configure_file:
                old_config = pickle.load(old_configure_file)

            # if speedy flag changes, overwriting is always needed for optimization
            if old_config["speedy"] != speedy:
                if conversion:
                    osim_data_overwrite = True

                    logger.info("       Speedy flag is different from last run, set osim_data_overwrite to TRUE.")

        else:
            osim_data_overwrite = True
            logger.info("       Configure of previous run does not exist, set osim_data_overwrite to TRUE.")

        logger.info("       All good now.")

        # save kwargs
        kwargs = {}
        kwargs["convert_steps"] = convert_steps
        kwargs["osim_data_overwrite"] = osim_data_overwrite
        kwargs["conversion"] = conversion
        kwargs["speedy"] = speedy

        # save the update kwargs to config.pkl in step 3 folder
        os.makedirs(output_folder + '/Step3_muscleKinetics', exist_ok = True)
        with open(output_folder + '/Step3_muscleKinetics/config.pkl', 'wb') as config_save:
            pickle.dump(kwargs, config_save)
    
    return osim_data_overwrite


def build_arg_parser():
    """Create the CLI ArgumentParser for O2MPipeline."""
    argparser = argparse.ArgumentParser(
        description=(
            "Convert an OpenSim model into a MuJoCo model with accurate muscle kinetics. "
            "Only works with OpenSim v4 models."
        )
    )
    argparser.add_argument("osim_file", type=str, help="Path to an OpenSim model OSIM file")
    argparser.add_argument(
        "geometry_folder",
        type=str,
        help="Path to the Geometry folder (by default uses folder of given OpenSim file)",
    )
    argparser.add_argument(
        "output_folder",
        type=str,
        help="Path to an output folder. The converted model will be saved here.",
    )

    argparser.add_argument(
        "--convert_steps",
        nargs="+",
        default=[1, 2, 3],
        help="Selected conversion steps, any subset of 1 2 3",
    )
    argparser.add_argument(
        "--muscle_list",
        nargs="+",
        type=str,
        default=None,
        help="Selected muscles for the conversion steps",
    )
    argparser.add_argument(
        "--osim_data_overwrite",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="If true, overwrite extracted Osim model state files",
    )
    argparser.add_argument(
        "--conversion",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="If true, perform the conversion functions of selected steps",
    )
    argparser.add_argument(
        "--validation",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="If true, perform the validation functions of selected steps",
    )
    argparser.add_argument(
        "--speedy",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="If true, reduce the number of checking nodes in optimization steps",
    )
    argparser.add_argument(
        "--generate_pdf",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="If true, generate a pdf report of the validation results",
    )
    argparser.add_argument(
        "--add_ground_geom",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="If true, a geom (of type plane) is added to the MuJoCo model as ground",
    )
    argparser.add_argument(
        "--treat_as_normal_path_point",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="If true, MovingPathPoints and ConditionalPathPoints will be treated as normal PathPoints",
    )
    argparser.add_argument(
        "--lenient_parsers",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="If true, log and skip unimplemented OpenSim plugin Force/Constraint tags instead of aborting",
    )
    argparser.add_argument(
        "--mjx",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="If true, emit an MJX/GPU-ready MJCF (<model>_mjx.xml) with primitive wraps and joint stability floors",
    )
    argparser.add_argument(
        "--mjx_integrator",
        type=str,
        default="implicitfast",
        help="MJX solver integrator when --mjx is enabled (implicitfast or Euler)",
    )
    argparser.add_argument(
        "--scale_set",
        type=str,
        default=None,
        help="Optional path to an OpenSim <ScaleSet>/<ScaleTool> XML file for subject-specific MJCF scaling",
    )
    argparser.add_argument(
        "--export_model_card",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="If true, export <model>_model_card.json with SHA-256 provenance and validation metrics",
    )
    return argparser


def parse_pipeline_args(argv=None):
    """Parse CLI arguments and normalize step and muscle lists."""
    args = build_arg_parser().parse_args(argv)
    args.convert_steps = parse_convert_steps(args.convert_steps)
    args.muscle_list = parse_muscle_list(args.muscle_list)
    return args


if __name__ == "__main__":
    parsed_args = parse_pipeline_args()
    O2MPipeline(**vars(parsed_args))
