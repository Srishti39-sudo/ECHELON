"""How fast does detection run on this machine, and does that keep up with a survey?

    .venv/bin/python tools/benchmark_edge.py                  # every variant present
    .venv/bin/python tools/benchmark_edge.py --runs 100 --threads 4
    .venv/bin/python tools/benchmark_edge.py --swath 75 --resolution 0.05 --speed 5

WHAT IS MEASURED
    For each detector variant that exists on disk -- torch (.pt through
    ultralytics), ONNX fp32, ONNX int8, ONNX fp16 -- with BOTH models loaded,
    because a survey runs known and anomaly over every tile:

    * cold start: seconds from a fresh interpreter to a loaded detector,
      including importing the runtime. Importing torch is a real part of what
      an edge box pays at boot, so it is counted, not excluded.
    * first tile: the first inference, separately, because CoreML and TensorRT
      compile or build on first use and a warm number would hide that.
    * warm latency per tile, p50 and p95 over --runs calls cycling over the
      real 640x640 tiles in samples/tiles. Each call is the full per-tile cost
      the survey engine pays: decode the JPEG, letterbox, both networks, NMS.
    * tiles per second (1 / mean warm latency).
    * peak resident memory of that process (getrusage ru_maxrss).
    * model file sizes.

    Every variant runs in its own subprocess, so torch's memory and threads do
    not leak into the ONNX numbers or the other way round, and cold start is
    really cold (the OS file cache is not, and that is not claimed).

    ONNX variants run pinned to the CPU provider, since the CPU is what every
    edge target has. If this onnxruntime build offers an accelerator provider
    (CoreML on a Mac, CUDA or TensorRT on a Jetson), fp32 is also run on it and
    reported as its own row.

WHAT THE THROUGHPUT NUMBERS ASSUME
    Translating tiles per second into survey terms needs a sonar geometry, and
    the defaults are stated rather than hidden:

    * swath W metres per side (default 100), so 2W across both channels;
    * r metres per pixel (default 0.1) across track, and the waterfall
      resampled to square pixels, so also r along track;
    * tiles of 640 px at stride 512, as survey_preparation cuts them.

    Then tiles per km of line = ceil((2W/r - 640) / 512 + 1) across, times
    1000 / (512 r) rows of tiles per km along. At a vessel speed v knots the
    line arrives at 1.852 v km/h, and the real-time factor is the km/h this
    machine can process divided by that. Above 1 it keeps up.

    The square-pixel assumption is conservative. At 100 m range sound needs
    about 0.13 s for the round trip, so the ping rate is capped near 7.5 Hz and
    at 4 knots successive pings are about 0.27 m apart. The along-track data is
    physically coarser than 0.1 m, and resampling it to 0.1 m manufactures more
    tiles per km than the sonar resolves. The real workload is lighter than
    this computes.

    Detection is not the whole pipeline. Tiling, deduplication, scoring and the
    export run once per survey over the detector's output and cost far less per
    tile than two convolutional networks, but they are not in this number.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TILE = 640
STRIDE = 512
KNOT_KMH = 1.852


# --- Child: one variant, one process ------------------------------------------

def _peak_rss_mb() -> float | None:
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / 1e6  # type: ignore[attr-defined]
        except Exception:
            return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Bytes on macOS, kilobytes on Linux.
    return peak / 1e6 if sys.platform == "darwin" else peak * 1024 / 1e6


def child(spec: dict) -> dict:
    started = time.perf_counter()
    if spec["kind"] == "torch":
        # ultralytics sets OMP_NUM_THREADS=1 on import unless it is already
        # set, so torch as the survey engine runs it uses ONE intra-op thread.
        # Setting it here, before that import, is what makes a torch row and
        # an ONNX row at the same --threads comparable. threads=0 leaves
        # ultralytics' own default in place, i.e. the product as shipped.
        if spec["threads"]:
            os.environ["OMP_NUM_THREADS"] = str(spec["threads"])
        from survey_hazard_map.hazard_detect import UltralyticsDetector

        detector = UltralyticsDetector([Path(p) for p in spec["paths"]], conf=spec["conf"])
        import torch
        runtime = {"torch": torch.__version__, "threads": torch.get_num_threads(),
                   "device": "cpu"}
    else:
        from survey_hazard_map.hazard_detect_onnx import OnnxDetector
        import onnxruntime

        detector = OnnxDetector([Path(p) for p in spec["paths"]], conf=spec["conf"],
                                threads=spec["threads"], providers=spec["providers"])
        runtime = {"onnxruntime": onnxruntime.__version__,
                   "threads": detector.threads or "onnxruntime default",
                   "providers": detector.providers, "provider": detector.provider}
    cold = time.perf_counter() - started

    tiles = [Path(t) for t in spec["tiles"]]
    t0 = time.perf_counter()
    detector(tiles[0])
    first = time.perf_counter() - t0
    for i in range(spec["warmup"]):
        detector(tiles[i % len(tiles)])

    times = []
    for i in range(spec["runs"]):
        tile = tiles[i % len(tiles)]
        t0 = time.perf_counter()
        detector(tile)
        times.append(time.perf_counter() - t0)

    times_ms = sorted(t * 1000 for t in times)
    mean = statistics.fmean(times)
    return {
        "cold_start_s": round(cold, 3),
        "first_tile_ms": round(first * 1000, 1),
        "warm_p50_ms": round(statistics.median(times_ms), 1),
        "warm_p95_ms": round(times_ms[min(len(times_ms) - 1, math.ceil(0.95 * len(times_ms)) - 1)], 1),
        "warm_mean_ms": round(mean * 1000, 1),
        "tiles_per_s": round(1 / mean, 2),
        "peak_rss_mb": (round(_peak_rss_mb(), 1) if _peak_rss_mb() is not None else None),
        "runtime": runtime,
        "runs": spec["runs"],
    }


# --- Parent -------------------------------------------------------------------

def _loadavg() -> list[float] | None:
    """1/5/15-minute load average. A benchmark on a busy machine is recorded as one."""
    try:
        return [round(v, 2) for v in os.getloadavg()]
    except (OSError, AttributeError):
        return None


def device_info() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(), "logical_cpus": os.cpu_count()}
    cpu = None
    try:
        if sys.platform == "darwin":
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
            # Apple silicon mixes performance and efficiency cores, and a
            # thread count above the performance-core count lands work on the
            # slow ones. Recorded so a reader can see why 8 threads can lose to 4.
            levels = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu",
                                     "hw.perflevel1.physicalcpu"],
                                    capture_output=True, text=True, timeout=5).stdout.split()
            if len(levels) == 2:
                info["performance_cores"], info["efficiency_cores"] = map(int, levels)
        elif Path("/proc/cpuinfo").exists():
            text = Path("/proc/cpuinfo").read_text()
            for key in ("model name", "Model", "Hardware"):
                for line in text.splitlines():
                    if line.startswith(key):
                        cpu = line.split(":", 1)[1].strip()
                        break
                if cpu:
                    break
    except Exception:
        pass
    info["cpu"] = cpu or platform.processor() or "unknown"
    try:
        import psutil
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["memory_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        pass
    if Path("/etc/nv_tegra_release").exists():
        info["jetson"] = Path("/etc/nv_tegra_release").read_text().strip()[:120]
    return info


def throughput(tiles_per_s: float, swath_m: float, res_m: float, speed_kn: float) -> dict:
    across_px = 2 * swath_m / res_m
    across = 1 if across_px <= TILE else math.ceil((across_px - TILE) / STRIDE) + 1
    rows_per_km = 1000 / (STRIDE * res_m)
    tiles_per_km = across * rows_per_km
    km_per_h = tiles_per_s * 3600 / tiles_per_km
    need_km_per_h = speed_kn * KNOT_KMH
    return {"tiles_across": across, "tiles_per_km": round(tiles_per_km, 1),
            "km_per_hour": round(km_per_h, 1), "required_km_per_hour": round(need_km_per_h, 2),
            "real_time_factor": round(km_per_h / need_km_per_h, 1)}


def variants(models_dir: Path, names: list[str]) -> list[tuple[str, str, list[Path]]]:
    """(label, kind, paths) for every variant whose files all exist."""
    found = []
    candidates = [("torch .pt", "torch", ".pt"), ("onnx fp32", "onnx", ".onnx"),
                  ("onnx int8", "onnx", ".int8.onnx"), ("onnx fp16", "onnx", ".fp16.onnx")]
    for label, kind, suffix in candidates:
        paths = [models_dir / f"{name}{suffix}" for name in names]
        if all(p.is_file() for p in paths):
            found.append((label, kind, paths))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--models", nargs="+", default=["known", "anomaly"])
    parser.add_argument("--tiles", type=Path, default=ROOT / "survey_hazard_map" / "samples" / "tiles")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4],
                        help="intra-op thread counts to run every CPU variant at, for torch "
                             "and onnxruntime alike (default: 1 4). 0 = each runtime's own "
                             "default, which for torch under ultralytics is 1")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--no-accelerator", action="store_true",
                        help="skip the extra fp32 row on an accelerator provider")
    parser.add_argument("--swath", type=float, default=100.0, help="metres per side")
    parser.add_argument("--resolution", type=float, default=0.1, help="metres per pixel")
    parser.add_argument("--speed", type=float, default=4.0, help="vessel speed, knots")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "edge_benchmark.json")
    args = parser.parse_args()

    if args.child:
        print(json.dumps(child(json.loads(args.child))))
        return 0

    tiles = sorted(str(p) for p in args.tiles.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not tiles:
        print(f"no tiles in {args.tiles}", file=sys.stderr)
        return 1

    plan: list[tuple[str, str, list[Path], list[str] | None, int]] = []
    for threads in args.threads:
        for label, kind, paths in variants(args.models_dir, args.models):
            tag = f"{threads}t" if threads else "default"
            if kind == "torch":
                plan.append((f"{label} cpu {tag}", kind, paths, None, threads))
            else:
                plan.append((f"{label} cpu {tag}", kind, paths, ["CPUExecutionProvider"], threads))
    try:
        import onnxruntime as ort
        accel = [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider",
                             "CoreMLExecutionProvider") if p in ort.get_available_providers()]
    except ImportError:
        accel = []
    fp32 = next((paths for label, kind, paths in variants(args.models_dir, args.models)
                 if label == "onnx fp32"), None)
    if accel and fp32 and not args.no_accelerator:
        short = accel[0].replace("ExecutionProvider", "")
        plan.append((f"onnx fp32 {short}", "onnx", fp32, [accel[0], "CPUExecutionProvider"], 0))
    if not plan:
        print(f"no model files found in {args.models_dir}", file=sys.stderr)
        return 1

    results = []
    for label, kind, paths, providers, threads in plan:
        spec = {"kind": kind, "paths": [str(p) for p in paths], "providers": providers,
                "threads": threads, "tiles": tiles, "runs": args.runs,
                "warmup": args.warmup, "conf": args.conf}
        print(f"running {label} ...", flush=True)
        load_before = _loadavg()
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child",
                               json.dumps(spec)], capture_output=True, text=True, cwd=str(ROOT))
        lines = [line for line in proc.stdout.strip().splitlines() if line.startswith("{")]
        if not lines:
            print(f"  {label} failed:\n{proc.stderr[-1500:]}", file=sys.stderr)
            results.append({"variant": label, "error": proc.stderr[-500:]})
            continue
        measured = json.loads(lines[-1])
        if proc.returncode != 0:
            # Seen intermittently on macOS: onnxruntime aborts in a mutex
            # during interpreter teardown, after every measurement has been
            # printed. The numbers are complete; the exit is recorded anyway.
            measured["exit_code"] = proc.returncode
            measured["exit_stderr"] = proc.stderr.strip()[-300:]
        measured.update({
            "load_average_before": load_before,
            "variant": label,
            "threads_requested": threads,
            "files": {p.name: round(p.stat().st_size / 1e6, 1) for p in paths},
            "model_mb": round(sum(p.stat().st_size for p in paths) / 1e6, 1),
            "survey": throughput(measured["tiles_per_s"], args.swath, args.resolution, args.speed),
        })
        results.append(measured)

    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": device_info(),
        "workload": {"models": args.models, "tiles": [Path(t).name for t in tiles],
                     "runs": args.runs, "warmup": args.warmup, "conf": args.conf,
                     "per_tile": "decode + letterbox + every model + NMS"},
        "survey_assumptions": {"swath_m_per_side": args.swath, "resolution_m_per_px": args.resolution,
                               "tile_px": TILE, "stride_px": STRIDE, "speed_knots": args.speed,
                               "square_pixels": True},
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    dev = report["device"]
    loads = [r["load_average_before"][0] for r in results if r.get("load_average_before")]
    if loads:
        print(f"\nLoad average (1 min) before each variant: {min(loads)}-{max(loads)} on "
              f"{dev['logical_cpus']} cores. Anything else running is in these numbers.")
    print(f"\nDevice: {dev['cpu']}, {dev.get('physical_cores', '?')} physical / "
          f"{dev['logical_cpus']} logical cores, {dev['platform']}")
    print(f"Workload: {' + '.join(args.models)} per tile, {len(tiles)} real 640x640 tiles, "
          f"{args.runs} timed runs")
    head = (f"{'variant':<26}{'MB':>6}{'cold s':>8}{'1st ms':>8}{'p50 ms':>8}{'p95 ms':>8}"
            f"{'tiles/s':>9}{'RSS MB':>8}{'km/h':>8}{'x RT':>8}  runtime")
    print(head)
    print("-" * len(head))
    for r in results:
        if "error" in r:
            print(f"{r['variant']:<26} failed")
            continue
        rt = r["runtime"]
        where = (f"torch {rt['torch']}, {rt['threads']} thr" if "torch" in rt
                 else f"ort {rt['onnxruntime']} {rt['provider'].replace('ExecutionProvider', '')}, "
                      f"threads {rt['threads']}")
        print(f"{r['variant']:<26}{r['model_mb']:>6}{r['cold_start_s']:>8}{r['first_tile_ms']:>8}"
              f"{r['warm_p50_ms']:>8}{r['warm_p95_ms']:>8}{r['tiles_per_s']:>9}"
              f"{r['peak_rss_mb'] or '-':>8}{r['survey']['km_per_hour']:>8}"
              f"{r['survey']['real_time_factor']:>8}  {where}")
    any_ok = next((r for r in results if "survey" in r), None)
    if any_ok:
        s = any_ok["survey"]
        print(f"\nSurvey: {args.swath:g} m per side at {args.resolution:g} m/px -> {s['tiles_across']} "
              f"tiles across, {s['tiles_per_km']} tiles per km of line; {args.speed:g} kn needs "
              f"{s['required_km_per_hour']} km/h. 'x RT' above 1 keeps up.")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
