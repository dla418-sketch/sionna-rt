
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Paths solver: Compute propagation paths"""

import drjit as dr

from .sb_candidate_generator import SBCandidateGenerator
from .image_method import ImageMethod
from .field_calculator import FieldCalculator
from .paths import Paths
from sionna.rt import Scene


class PathSolver:
    # pylint: disable=line-too-long
    r"""
    Class implementing a path solver

    A path solver computes propagation paths between the antennas of
    all transmitters and receivers in the a scene.
    For each propagation path :math:`i`, the corresponding channel coefficient
    :math:`a_i` and delay :math:`\tau_i`, the
    angles of departure :math:`(\theta_{\text{T},i}, \varphi_{\text{T},i})`
    and arrival :math:`(\theta_{\text{R},i}, \varphi_{\text{R},i})`, as
    well as the Doppler shifts :math:`f_{\Delta, i}` are computed.
    For more detail, see :eq:`H_final`.
    This path solver currently supports line-of-sigth, specular and diffuse
    reflection, as well as refraction. Paths can consist of any of these interaction
    types, in any order.
    Different propagation phenomena can be individually enabled/disabled.

    This solver assumes that materials are thin enough that their effect on
    transmitted rays (i.e., rays that traverse the materials through double
    refraction) is negligible. Rays are traced without angular deflection, and
    objects like walls should be modeled as single flat surfaces having an
    attached radio material that accounts for their
    :attr:`~sionna.rt.RadioMaterial.thickness`.
    This approach may be inaccurate for very thick objects.
    The figure below illustrates this model, where :math:`E_i` is the
    incident electric field, :math:`E_r` is the reflected field and :math:`E_t` is
    the transmitted field. The Jones matrices, :math:`\mathbf{R}(d)` and
    :math:`\mathbf{T}(d)`, represent the effects of reflection and transmission,
    respectively, and depend on the slab thickness, :math:`d`.

    .. figure:: ../figures/transmission_model.png
        :width: 80%
        :align: center

    If synthetic arrays are used (``synthetic_array`` is `True`), transmitters
    and receivers are modelled as if they had a single antenna located at their
    :attr:`~sionna.rt.RadioDevice.position`. The channel responses for each
    individual antenna of the arrays are then computed "synthetically" by applying
    appropriate phase shifts. This reduces the complexity significantly
    for large arrays. Time evolution of the channel coefficients can be simulated with
    using :meth:`~sionna.rt.Paths.cir` and :meth:`~sionna.rt.Paths.cfr` methods  of the returned
    :class:`~sionna.rt.Paths` object.

    Example
    -------
    .. code-block:: python

        import sionna
        from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver
        import mitsuba as mi

        # Load example scene
        scene = load_scene(sionna.rt.scene.munich)

        # Configure antenna array for all transmitters
        scene.tx_array = PlanarArray(num_rows=8,
                                    num_cols=2,
                                    vertical_spacing=0.7,
                                    horizontal_spacing=0.5,
                                    pattern="tr38901",
                                    polarization="VH")

        # Configure antenna array for all receivers
        scene.rx_array = PlanarArray(num_rows=1,
                                    num_cols=1,
                                    vertical_spacing=0.5,
                                    horizontal_spacing=0.5,
                                    pattern="dipole",
                                    polarization="cross")

        # Create transmitter
        tx = Transmitter(name="tx",
                        position=mi.Point3f(8.5,21,27),
                        orientation=mi.Point3f(0,0,0))
        scene.add(tx)

        # Create a receiver
        rx = Receiver(name="rx",
                    position=mi.Point3f(45,90,1.5),
                    orientation=mi.Point3f(0,0,0))
        scene.add(rx)

        # TX points towards RX
        tx.look_at(rx)

        # Compute paths
        solver = PathSolver()
        paths = solver(scene)

        # Open preview showing paths
        scene.preview(paths=paths, resolution=[1000,600], clip_at=15.)

    .. figure:: ../figures/paths_preview.png
        :align: center
    """

    def __init__(self):

        # Instantiate the Candidate Generator
        self._candidate_generator = SBCandidateGenerator()
        # Instantiate the Image Method solver
        self._image_method = ImageMethod()
        # Instantiate the Field Calculator
        self._field_calculator = FieldCalculator()

    @property
    def loop_mode(self):
        # pylint: disable=line-too-long
        r"""Get/set the Dr.Jit mode used to evaluate the loops that implement
        the solver. Should be one of "evaluated" or "symbolic". Symbolic mode
        (default) is the fastest one but does not support automatic
        differentiation.
        For more details, see the `corresponding Dr.Jit documentation <https://drjit.readthedocs.io/en/latest/cflow.html#sym-eval>`_.

        :type: "evaluated" | "symbolic"
        """
        return self._field_calculator.loop_mode

    @loop_mode.setter
    def loop_mode(self, mode):
        if mode not in ("evaluated", "symbolic"):
            raise ValueError("Invalid loop mode. Must be either 'evaluated'"
                             " or 'symbolic'")
        self._image_method.loop_mode = mode
        self._field_calculator.loop_mode = mode

    def __call__(self,
                 scene: Scene,
                 max_depth: int = 3,
                 max_num_paths_per_src: int = 1000000,
                 samples_per_src: int = 1000000,
                 synthetic_array: bool = True,
                 los: bool = True,
                 specular_reflection: bool = True,
                 diffuse_reflection: bool = False,
                 refraction: bool = True,
                 diffraction: bool = False,
                 edge_diffraction: bool = False,
                 diffraction_lit_region: bool = True,
                 seed: int = 42) -> Paths:
        # pylint: disable=line-too-long
        r"""
        Executes the solver

        :param scene: Scene for which to compute paths
        :param max_depth: Maximum depth
        :param max_num_paths_per_src: Maximum number of paths per source
        :param samples_per_src: Number of samples per source
        :param synthetic_array: If set to `True` (default), then the antenna arrays are applied synthetically
        :param los: Enable line-of-sight paths
        :param specular_reflection: Enables specular reflection
        :param diffuse_reflection: Enables diffuse reflection
        :param refraction: Enables refraction
        :param diffraction: Enables diffraction
        :param edge_diffraction: Enables diffraction on free floating edges
        :param diffraction_lit_region: Enables diffraction in the lit region
        :param seed: Seed

        :return: Computed paths
        """

        # Check that the scene is all set for simulations
        scene.all_set(radio_map=False)

        # Generates sources positions and orientations
        src_positions, src_orientations, rel_ant_positions_tx, tx_velocities =\
                                            scene.sources(synthetic_array, True)
        tgt_positions, tgt_orientations, rel_ant_positions_rx, rx_velocities =\
                                            scene.targets(synthetic_array, True)

        # Trace paths and compute channel impulse responses
        src_antenna_patterns = scene.tx_array.antenna_pattern.patterns
        tgt_antenna_patterns = scene.rx_array.antenna_pattern.patterns
        dr.make_opaque(src_positions, tgt_positions, src_orientations,
                       tgt_orientations)

        # Generate candidates
        paths_buffer = self._candidate_generator(
            mi_scene=scene.mi_scene,
            src_positions=src_positions,
            tgt_positions=tgt_positions,
            samples_per_src=samples_per_src,
            max_num_paths_per_src=max_num_paths_per_src,
            max_depth=max_depth,
            los=los,
            specular_reflection=specular_reflection,
            diffuse_reflection=diffuse_reflection,
            refraction=refraction,
            diffraction=diffraction,
            edge_diffraction=edge_diffraction,
            seed=seed
        )

        paths_buffer.schedule()
        dr.eval()

        # Shrink the paths buffer to fit the number of paths effectively found
        paths_buffer.shrink()

        # Detach the paths geometry to avoid differentiation through the
        # candidate generator
        paths_buffer.detach_geometry()

        # Solve specular chains and suffixes
        paths_buffer = self._image_method(
            scene=scene.mi_scene,
            paths=paths_buffer,
            src_positions=src_positions,
            tgt_positions=tgt_positions,
            diffraction=diffraction,
            diffraction_lit_region=diffraction_lit_region
        )

        # Compute channel coefficients and delays
        paths_buffer = self._field_calculator(
            wavelength=scene.wavelength,
            paths=paths_buffer,
            samples_per_src=samples_per_src,
            diffraction=diffraction,
            src_positions=src_positions,
            tgt_positions=tgt_positions,
            src_orientations=src_orientations,
            tgt_orientations=tgt_orientations,
            src_antenna_patterns=src_antenna_patterns,
            tgt_antenna_patterns=tgt_antenna_patterns
        )

        # Discard invalid paths
        paths_buffer.discard_invalid()

        # Build the path object
        paths = Paths(scene, src_positions, tgt_positions, tx_velocities,
                      rx_velocities, synthetic_array, paths_buffer,
                      rel_ant_positions_tx, rel_ant_positions_rx)

        return paths
