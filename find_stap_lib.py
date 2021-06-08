import subprocess
from bcc import USDT


PROGRAM = "python main.py"
STAP_PROVIDER = "numba_stap"


def main():
    pid = pidof(PROGRAM)
    stap_lib_location = find_stap_lib(pid)
    print(stap_lib_location.bin_path.decode("ascii"))


def find_stap_lib(pid):
    reader = USDT(pid=pid)
    location = None
    for probe in reader.enumerate_probes():
        if probe.provider.decode("ascii") == STAP_PROVIDER:
            location = probe.get_location(0)
    return location


def pidof(program):
    pgrep = subprocess.Popen(
        ["pgrep", "-f", program],
        stdout=subprocess.PIPE,
        shell=False,
    )
    result = pgrep.communicate()[0]
    return int(result.strip().decode('ascii'))


if __name__ == "__main__":
    main()
