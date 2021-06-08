This is a demonstration of using SystemTap probes in numba code.

## Setup:
Install [BPF tools](https://github.com/iovisor/bcc).

Install [libstapsdt](https://github.com/sthima/libstapsdt).

Make a virtualenv if you like. Run `pip install -r requirements.txt`.

If you can run `python main.py`, things are working.

## Running the demo

In one terminal session, run `python main.py`. There will be a long pause at
`iteration 1` while the numba JIT compiler works:

```
-> % python main.py
iteration 1
iteration 2
iteration 3
iteration 4
iteration 5
iteration 6
iteration 7
iteration 8
iteration 9
iteration 10
iteration 11
iteration 12
iteration 13
[...]
```

Now, open a second terminal session. We will trace the program now:
```
-> % export STAPLIB=$(python find_stap_lib.py)
-> % sudo bpftrace -e "usdt:$STAPLIB:numba_stap:* { printf(\"%s %s\n\", probe, ustack) } "
Attaching 11 probes...
usdt:/proc/17593/root/tmp/numba_stap-KGBdiu.so:numba_stap:find_runs
        0x7f558d139468
        0x7f558c29d073
        0x7f558c292593
        0x7f558bfd2198
        0x7f558bfd3957
        call_cfunc(Dispatcher*, _object*, _object*, _object*, _object*)+76
        0x94f380

usdt:/proc/17593/root/tmp/numba_stap-KGBdiu.so:numba_stap:label_clusters
        0x7f558d139472
        0x7f558c29904c

usdt:/proc/17593/root/tmp/numba_stap-KGBdiu.so:numba_stap:hotspot_2d_inner
        0x7f558d139440
        0x7f558c292101
        0x7f558bfd23c4
        0x7f558bfd3957
        call_cfunc(Dispatcher*, _object*, _object*, _object*, _object*)+76
        0x94f380

usdt:/proc/17593/root/tmp/numba_stap-KGBdiu.so:numba_stap:make_points_nonzero
        0x7f558d139459
        0x7f55937ce02e
        pyobject_dtor+0

usdt:/proc/17593/root/tmp/numba_stap-KGBdiu.so:numba_stap:quantize_points
        0x7f558d13945e
        0x7f55937bd099
        nrt_internal_dtor_safe+0

usdt:/proc/17593/root/tmp/numba_stap-KGBdiu.so:numba_stap:sort_order_2d
        0x7f558d139463
        0x7f558c2a1045
        0x7f558c292330
        0x7f558bfd23c4
        0x7f558bfd3957
        call_cfunc(Dispatcher*, _object*, _object*, _object*, _object*)+76
        0x94f380
[...]
```

### Loading events into perf

Tracepoint data can be loaded into perf with this one-liner:
```
-> % sudo perf probe -x $(python find_stap_lib.py) -a '*'
```
