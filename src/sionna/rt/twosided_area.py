#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Custom Mitsuba emitter to turn a one-sided area emitter into a two-sided one.
"""

from __future__ import annotations

import drjit as dr
import mitsuba as mi


class TwosidedAreaEmitter(mi.Emitter):
    """Custom Mitsuba emitter to turn a one-sided area emitter into a
    two-sided one.

    This is used for visualization of mesh-based radio maps, and hasn't been
    thoroughly tested for correctness.
    """

    def __init__(self, props):
        super().__init__(props)

        self.nested: mi.Emitter | None = props.get("nested", None)
        if not isinstance(self.nested, mi.Emitter):
            raise ValueError("TwosidedAreaEmitter: must provide an emitter with"
                             " name=\"nested\".")

        self.m_flags = self.nested.m_flags
        self.m_needs_sample_2 = self.nested.m_needs_sample_2
        self.m_needs_sample_3 = self.nested.m_needs_sample_3


    def sample_ray(self, time: mi.Float, sample1: mi.Float,
                   sample2: mi.Point2f, sample3: mi.Point2f,
                   active: mi.Mask) -> tuple[mi.Ray3f, mi.Spectrum]:
        # Equal probability of choosing either side
        flip = sample1 > 0.5
        # Reuse sample1
        sample1 = (sample1 - 0.5) * 2

        ray, weight = self.nested.sample_ray(time, sample1, sample2, sample3,
                                             active)
        ray.d = dr.select(flip, -ray.d, ray.d)
        return ray, weight


    def sample_direction(self, ref: mi.Interaction3f, sample: mi.Point2f,
                         active: mi.Mask) -> tuple[mi.DirectionSample3f,
                                                   mi.Spectrum]:
        # Note: this wasn't fully tested for correctness.
        ds, weight = self.nested.sample_direction(ref, sample, active)

        # If we ended up with zero radiance because we were on the wrong side,
        # we try reflecting the `ref` position to the other side along the
        # sampled direction and hope it will yield a correct result.
        # It may not work for more sophisticated sampling methods.
        wrong_side = active & (dr.dot(ds.n, ds.d) >= 0)
        ref.p = dr.select(wrong_side, ref.p + 2 * ds.dist * ds.d, ref.p)
        ds2, weight2 = self.nested.sample_direction(ref, sample,
                                                    active & wrong_side)

        return dr.select(wrong_side, ds2, ds), \
               dr.select(wrong_side, weight2, weight)

    def pdf_direction(self, ref: mi.Interaction3f, ds: mi.DirectionSample3f,
                      active: mi.Mask) -> mi.Float:
        wrong_side = dr.dot(ds.n, ds.d) >= 0
        ds.d = dr.select(wrong_side, -ds.d, ds.d)
        return self.nested.pdf_direction(ref, ds, active)

    def eval_direction(self, ref: mi.Interaction3f, ds: mi.DirectionSample3f,
                       active: mi.Mask) -> mi.Spectrum:
        wrong_side = dr.dot(ds.n, ds.d) >= 0
        ds.d = dr.select(wrong_side, -ds.d, ds.d)
        return self.nested.eval_direction(ref, ds, active)

    def eval(self, si: mi.SurfaceInteraction3f, active: mi.Mask) -> mi.Spectrum:
        # Note: `si.wi` points away from the surface, in local coordinates.
        wrong_side = si.wi.z <= 0
        si.wi = dr.select(wrong_side, -si.wi, si.wi)
        return self.nested.eval(si, active)

    def sample_position(self, time: mi.Float, sample: mi.Point2f,
                        active: mi.Mask) -> tuple[mi.PositionSample3f,
                                                  mi.Float]:
        return self.nested.sample_position(time, sample, active)

    def pdf_position(self,
                     ps: mi.PositionSample3f,
                     active: mi.Mask) -> mi.Float:
        return self.nested.pdf_position(ps, active)

    def sample_wavelengths(self, *args, **kwargs):
        return self.nested.sample_wavelengths(*args, **kwargs)

    def pdf_wavelengths(self, *args, **kwargs):
        return self.nested.pdf_wavelengths(*args, **kwargs)

    def bbox(self):
        return self.nested.bbox()

    def to_string(self):
        return f"TwosidedAreaEmitter[{repr(self.nested)}]"

    def traverse(self, *args, **kwargs):
        self.nested.traverse(*args, **kwargs)

    def parameters_changed(self, *args, **kwargs):
        self.nested.parameters_changed(*args, **kwargs)



# Register this custom Emitter plugin
mi.register_emitter("twosided_area",
                    lambda props: TwosidedAreaEmitter(props=props))
