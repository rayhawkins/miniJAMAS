"""
    PyJAMAS is Just A More Awesome Siesta
    Copyright (C) 2018  Rodrigo Fernandez-Gonzalez (rodrigo.fernandez.gonzalez@utoronto.ca)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
from typing import Optional, Tuple
import numpy
import pyjamas.pjscore as pjscore


class RCallback:
    def __init__(self, ui: pjscore.PyJAMAS):
        """

        :type ui: pyjamas.PyJAMAS
        """
        super(RCallback, self).__init__()
        self.pjs = ui

    def get_coordinate_bounds(self, coordinates: Optional[numpy.ndarray] = None, margin_size: int = 0) -> (int, int, int, int):
        """
        :param coordinates: an array of coordinates OR an integer indicating a polygon index in self.pjs.polylines.
        :param margin_size:
        :return:
        """
        if coordinates.size == 1:
            index = numpy.min(coordinates, axis=0)
            index = int(index)
            minx, maxx, miny, maxy = None, None, None, None
            for slice_num in range(self.pjs.n_frames):
                if (self.pjs.polylines[slice_num] != []) and (len(self.pjs.polylines[slice_num]) > index):
                    polygon = self.pjs.polylines[slice_num][index]
                    bound = polygon.boundingRect()
                    if (minx is None) or (bound.topLeft().x() < minx):
                        minx = bound.topLeft().x()
                    if (miny is None) or (bound.topLeft().y() < miny):
                        miny = bound.topLeft().y()
                    if (maxy is None) or (bound.bottomRight().y() > maxy):
                        maxy = bound.bottomRight().y()
                    if (maxx is None) or (bound.bottomRight().x() > maxx):
                        maxx = bound.bottomRight().x()

        else:
            # This is here mainly to crop around non-rectangular polylines.
            minx, miny = numpy.min(coordinates, axis=0)
            maxx, maxy = numpy.max(coordinates, axis=0)

        # Make sure you are working with integers to prevent errors when slicing the original image.
        minx, miny, maxx, maxy = int(minx), int(miny), int(maxx), int(maxy)

        # Adjust for margin
        minx, miny, maxx, maxy = minx - margin_size, miny - margin_size, maxx + margin_size, maxy + margin_size

        # Check if within the image bounds, and set to max/min possible value if not
        if maxx >= self.pjs.width:
            maxx = self.pjs.width - 1
        if maxy >= self.pjs.height:
            maxy = self.pjs.height - 1
        if minx < 0:
            minx = 0
        if miny < 0:
            miny = 0

        return minx, miny, maxx, maxy

    def generate_ROI_filename(self, x_range: Tuple[int, int], y_range: Tuple[int, int], z_range: Tuple[int, int], extension: str, relative: bool = False) -> str:
        _, fname = os.path.split(self.pjs.filename)
        fname, _ = os.path.splitext(fname)

        fname += '_X' + str(x_range[0]) + '_' + str(x_range[1]) + '_Y' + str(y_range[0]) + '_' + str(
            y_range[1]) + '_Z' + str(z_range[0]) + '_' + str(z_range[1]) + extension

        return os.path.join(self.pjs.cwd, fname) if not relative else fname

