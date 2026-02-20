#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Module implementing radio materials for the Sionna RT"""

from .radio_material_base import RadioMaterialBase
from .radio_material import RadioMaterial
from .itu_material import ITURadioMaterial
from .scattering_pattern import register_scattering_pattern, \
                                scattering_pattern_registry, \
                                ScatteringPattern, \
                                LambertianPattern, \
                                BackscatteringPattern, \
                                DirectivePattern
