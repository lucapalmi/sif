# Copyright (C) 2026 Luca Palmieri
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of sif. See COPYING for the full license text.

# pysif.finders is compiled into _pysif, which registers it in sys.modules as
# pysif loads, so an import finds it there and never reads this file. The
# file is for editors, which find finders.pyi beside it and warn about a stub
# with no module behind it. Should anything import it all the same, it hands
# over the compiled module.
import sys

from . import _pysif

sys.modules[__name__] = _pysif.finders
