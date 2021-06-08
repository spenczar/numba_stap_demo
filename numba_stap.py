from ctypes import c_char_p, c_int
import stapsdt

_numba_provider = stapsdt.providerInit(c_char_p("numba_stap".encode("ascii")))


def create_probe(name):
    probe = stapsdt.providerAddProbe(
        _numba_provider,
        c_char_p(name.encode("ascii")),
        c_int(0),
    )
    enable_probes()
    return probe


def enable_probes():
    code = stapsdt.providerLoad(_numba_provider)
    if code != 0:
        raise Exception(f"provider load, error code {code}")


def disable_probes():
    code = stapsdt.providerUnload(_numba_provider)
    if code != 0:
        raise Exception(f"provider unload, error code {code}")


fire_probe = stapsdt.probeFire
