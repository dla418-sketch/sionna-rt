#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Material properties from Section 2.1.4 of recommendation ITU-R P2040"""

import drjit as dr
import mitsuba as mi
from typing import Tuple


# Data structure storing the properties from Table 3 of ITU-R P2040.
# A material can be mapped to multiple frequency ranges, each with a
# different set of parameters.
#
# Structure :
#   material_name: { (min_freq [GHz], max_freq [GHz]): (a, b, c, d) }
ITU_MATERIALS_PROPERTIES = {
    "concrete"          :   { (1., 100.)    :   (5.24, 0.0, 0.0462, 0.7822) },

    "brick"             :   { (1., 40.)     :   (3.91, 0.0, 0.0238, 0.16)   },

    "plasterboard"      :   { (1., 100.)    :   (2.73, 0.0, 0.0085, 0.9395) },

    "wood"              :   { (0.001, 100.) :   (1.99, 0.0, 0.0047, 1.0718) },

    "glass"             :   { (0.1, 100.)   :   (6.31, 0.0, 0.0036, 1.3394),
                              (220., 450.)  :   (5.79, 0.0, 0.0004, 1.658)  },

    "ceiling_board"     :   { (1.0, 100.)   :   (1.48, 0.0, 0.0011, 1.0750),
                              (220., 450.)  :   (1.52, 0.0, 0.0029, 1.029)  },

    "chipboard"         :   { (1.0, 100.)   :   (2.58, 0.0, 0.0217, 0.7800) },

    "plywood"           :   { (1.0, 40.)    :   (2.71, 0.0, 0.33, 0.0) },

    "marble"            :   { (1.0, 60.)    :   (7.074, 0.0, 0.0055, 0.9262) },

    "floorboard"        :   { (50., 100.)   :   (3.66, 0.0, 0.0044, 1.3515) },

    "metal"             :   { (1.0, 100.)   :   (1.0, 0.0, 1e7, 0.0) },

    "very_dry_ground"   :   { (1.0, 10.)    :   (3.0, 0.0, 0.00015, 2.52) },

    "medium_dry_ground" :   { (1.0, 10.)    :   (15., -0.1, 0.035, 1.63) },

    "wet_ground"        :   { (1.0, 10.)    :   (30., -0.4, 0.15, 1.30) }
}


def itu_material(name: str, f: mi.Float) -> Tuple[mi.Float, mi.Float]:
    r"""
    Evaluates the real component of the relative permittivity and the
    conductivity [S/m] of the ITU material `name` for the frequency `f` [Hz]

    Implements model from Section 2.1.4 of recommendation ITU-R P2040.

    :param name: Name of the ITU material to evaluate.
        Must be a key of `ITU_MATERIALS_PROPERTIES`.
    :param f: Frequency [Hz]

    :return: Real component of the relative permittivity and conductivity [S/m]
    """

    if name not in ITU_MATERIALS_PROPERTIES:
        raise ValueError(f"Unknown ITU material '{name}'")
    props = ITU_MATERIALS_PROPERTIES[name]

    f_ghz = f/1e9

    # Extract the properties to use according to the frequency
    # If the frequency is in none of the valid ranges, an exception is raised
    valid_freq = False
    for f_ranges, params in props.items():
        if f_ranges[0] <= f_ghz <= f_ranges[1]:
            a, b, c, d = params
            valid_freq = True
            break
    if not valid_freq:
        raise ValueError(f"Properties of ITU material '{name}' are not defined"
                         " for this frequency")

    # Evaluate the material properties
    eta_r = a * dr.power(f_ghz, b)
    sigma = c * dr.power(f_ghz, d)

    return eta_r, sigma
