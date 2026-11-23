#!/usr/bin/env python3
"""procmap.py with the source clips read from gc25d_v10_fxcond (the paper test clips' meta / latents live there;
procmap's default data root gc25d_v10 only holds part of them). Same CLI as procmap.py."""

import os as _mos

_MROOT = _mos.path.abspath(_mos.path.dirname(__file__))
while _MROOT != _mos.path.dirname(_MROOT) and not _mos.path.isdir(_mos.path.join(_MROOT, "code", "gc25d")):
    _MROOT = _mos.path.dirname(_MROOT)
MALIANG_ROOT = _mos.environ.get("MALIANG_ROOT") or _MROOT
import sys
RT = MALIANG_ROOT
sys.path[:0] = [RT + "/code/gc25d"]
import procmap as PM
PM.DATA = RT + "/data/gc25d_v10_fxcond"
PM.load_source.__defaults__ = (PM.DATA,)
if __name__ == "__main__":
    sys.exit(PM.main())
