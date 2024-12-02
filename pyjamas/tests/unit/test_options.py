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
import os.path

# tests for options menu
# for coverage, run:
# coverage run -m pytest
# or if you want to include branches:
# coverage run --branch -m pytest
# followed by:
# coverage report -i

from pyjamas.pjscore import PyJAMAS
from pyjamas.rimage.rimcore import rimage

PyJAMAS_FIXTURE: PyJAMAS = PyJAMAS()


def test_cbSetBrushSize():
    old_brush_sz = PyJAMAS_FIXTURE.brush_size

    assert PyJAMAS_FIXTURE.options.cbSetBrushSize(34)
    assert not PyJAMAS_FIXTURE.options.cbSetBrushSize(-2)
    assert PyJAMAS_FIXTURE.options.cbSetBrushSize(old_brush_sz)

def test_cbDisplayFiducialIDs():
    assert PyJAMAS_FIXTURE.options.cbDisplayFiducialIDs()
    assert PyJAMAS_FIXTURE.options.cbDisplayFiducialIDs()

def test_cbFramesPerSec():
    old_fps = PyJAMAS_FIXTURE.fps

    assert PyJAMAS_FIXTURE.options.cbFramesPerSec(34)
    assert not PyJAMAS_FIXTURE.options.cbFramesPerSec(-2)
    assert PyJAMAS_FIXTURE.options.cbSetBrushSize(old_fps)

def test_cbSetCWD():
    old_cwd = PyJAMAS_FIXTURE.cwd

    assert PyJAMAS_FIXTURE.options.cbSetCWD(os.path.abspath('.'))
    assert not PyJAMAS_FIXTURE.options.cbSetCWD('-1')
    assert PyJAMAS_FIXTURE.options.cbSetCWD(old_cwd)

def test_cbSetLivewireShortestPathFunction():
    fn_names = tuple(rimage.livewire_shortest_path_fns.keys())
    assert PyJAMAS_FIXTURE.options.cbSetLivewireShortestPathFunction(rimage.livewire_shortest_path_fns[fn_names[-1]])

    assert PyJAMAS_FIXTURE.options.cbSetLivewireShortestPathFunction(rimage.livewire_shortest_path_fns[fn_names[0]])

