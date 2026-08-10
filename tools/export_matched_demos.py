#!/usr/bin/env python3
"""Generate case-matched RelFx website audio with the paper's ITO setup."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


SAMPLE_RATE = 44_100
PROCESSOR_ORDER = [
    "eq",
    "multiband_comp",
    "imager",
    "gain",
    "distortion",
    "delay",
    "limiter",
]
ACTIVATION_ORDER = [
    "eq",
    "distortion",
    "multiband_comp",
    "gain",
    "imager",
    "limiter",
    "delay",
    "reverb",
]
PROCESSOR_LABELS = {
    "eq": "EQ",
    "multiband_comp": "Multiband Compressor",
    "imager": "Stereo Imager",
    "gain": "Gain",
    "distortion": "Distortion",
    "delay": "Delay",
    "limiter": "Limiter",
}
METHODS = [
    {
        "id": "fxencoderpp",
        "label": "Fx-Encoder++",
        "model": "fxencoderpp",
        "embed_mode": "dry_wet",
        "kind": "baseline",
        "description": "Single-input benchmark baseline",
    },
    {
        "id": "relfx_self_ref",
        "label": "RelFx (Self-ref)",
        "model": "relfx",
        "embed_mode": "wet_wet",
        "kind": "relfx",
        "description": "No dry reference required",
    },
    {
        "id": "relfx_standard",
        "label": "RelFx (Standard)",
        "model": "relfx",
        "embed_mode": "dry_wet",
        "kind": "relfx",
        "description": "Source-anchored relative encoding",
    },
    {
        "id": "relfx_oracle",
        "label": "RelFx (Oracle)",
        "model": "relfx",
        "embed_mode": "cross_dry_wet",
        "kind": "oracle",
        "description": "Uses the ground-truth dry reference for analysis",
    },
]
MAX_OPTIMIZATION_ATTEMPTS = 8


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Optimize matched website examples on fixed MUSDB18 triplets using "
            "the ISMIR 2026 Fx style-transfer procedure."
        )
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--triplets-dir",
        type=Path,
        default=Path("/data2/08_parameter_matching/triplets/musdb18"),
    )
    parser.add_argument(
        "--relfx-checkpoint",
        type=Path,
        default=(
            project_root
            / "relfx-ismir2026-hf/relfx-ismir2026-epoch199.pt"
        ),
    )
    parser.add_argument(
        "--fxencoderpp-checkpoint",
        type=Path,
        default=(
            project_root
            / "checkpoints/fxencoder++/fxenc_plusplus_default.pt"
        ),
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=script_dir / "selected_cases.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir.parent / "audio/matched",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=script_dir.parent / "data/demos.json",
    )
    parser.add_argument(
        "--state-output",
        type=Path,
        default=script_dir.parent / "data/optimization-state.json",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[method["id"] for method in METHODS],
        default=[method["id"] for method in METHODS],
    )
    parser.add_argument(
        "--force-methods",
        nargs="+",
        choices=[method["id"] for method in METHODS],
        default=[],
        help="Re-run selected methods while retaining other completed outputs.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--n-iters", type=int, default=200)
    parser.add_argument("--num-restarts", type=int, default=7)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--es-patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3182026)
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_audio(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(
        path, dtype="float32", always_2d=True
    )
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sample_rate}: {path}")
    return audio.T


def write_audio(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.T, SAMPLE_RATE, subtype="PCM_16")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_stats(audio: np.ndarray) -> dict[str, float]:
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    rms_dbfs = float(20.0 * np.log10(max(rms, 1e-12)))
    return {
        "duration_seconds": round(audio.shape[-1] / SAMPLE_RATE, 3),
        "peak": round(peak, 6),
        "rms_dbfs": round(rms_dbfs, 3),
    }


def active_processors(params_meta: dict[str, Any]) -> list[str]:
    activation = params_meta["activate"]
    return [
        processor
        for processor in PROCESSOR_ORDER
        if activation[ACTIVATION_ORDER.index(processor)] > 0.5
    ]


def physical_value(
    normalized: float, value_range: tuple[float, float]
) -> float:
    low, high = value_range
    return float(low + normalized * (high - low))


def format_frequency(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.1f} kHz"
    return f"{value:.0f} Hz"


def format_key_parameters(
    fx_chain: Any, params_meta: dict[str, Any]
) -> list[dict[str, Any]]:
    values = params_meta["nn_param"]
    readable: dict[str, dict[str, float]] = {}
    processors = fx_chain.chain.fx_chain.fx_processors

    for processor_name in PROCESSOR_ORDER:
        processor = processors[processor_name]
        processor_index = fx_chain.chain.fx_chain.fx_indices[processor_name]
        start, _ = fx_chain.chain.fx_chain.param_range[processor_index]
        readable[processor_name] = {
            name: physical_value(float(values[start + offset]), value_range)
            for offset, (name, value_range) in enumerate(
                processor.param_ranges.items()
            )
        }

    result: list[dict[str, Any]] = []
    for processor_name in active_processors(params_meta):
        parameters = readable[processor_name]
        summary: list[str] = []

        if processor_name == "eq":
            bands = [
                ("Low shelf", "low_shelf_gain_db", "low_shelf_cutoff_freq"),
                ("Low-mid", "band0_gain_db", "band0_cutoff_freq"),
                ("Mid", "band1_gain_db", "band1_cutoff_freq"),
                ("High-mid", "band2_gain_db", "band2_cutoff_freq"),
                ("High", "band3_gain_db", "band3_cutoff_freq"),
                ("High shelf", "high_shelf_gain_db", "high_shelf_cutoff_freq"),
            ]
            strongest = sorted(
                bands,
                key=lambda item: abs(parameters[item[1]]),
                reverse=True,
            )[:2]
            summary = [
                (
                    f"{label}: {parameters[gain_key]:+.1f} dB at "
                    f"{format_frequency(parameters[freq_key])}"
                )
                for label, gain_key, freq_key in strongest
            ]
        elif processor_name == "multiband_comp":
            summary = [
                f"Parallel mix: {parameters['parallel_weight_factor'] * 100:.0f}%",
                (
                    "Ratios (L/M/H): "
                    f"{parameters['low_shelf_comp_ratio']:.1f}:1 / "
                    f"{parameters['mid_band_comp_ratio']:.1f}:1 / "
                    f"{parameters['high_shelf_comp_ratio']:.1f}:1"
                ),
            ]
        elif processor_name == "imager":
            summary = [f"Width: {parameters['width'] * 100:.0f}%"]
        elif processor_name == "gain":
            summary = [f"Gain: {parameters['gain_db']:+.1f} dB"]
        elif processor_name == "distortion":
            summary = [f"Drive: {parameters['drive_db']:.1f} dB"]
        elif processor_name == "delay":
            delay_ms = parameters["delay_samples"] / SAMPLE_RATE * 1000.0
            summary = [
                f"Time: {delay_ms:.0f} ms",
                f"Wet: {parameters['wet'] * 100:.0f}%",
            ]
        elif processor_name == "limiter":
            summary = [
                f"Threshold: {parameters['threshold']:.1f} dB",
                f"Attack: {parameters['at']:.1f} ms",
                f"Release: {parameters['rt']:.0f} ms",
            ]

        result.append(
            {
                "id": processor_name,
                "label": PROCESSOR_LABELS[processor_name],
                "summary": summary,
            }
        )
    return result


def copy_shared_audio(
    sample_dir: Path, output_dir: Path
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    sources = {
        "source_dry": sample_dir / "clean.wav",
        "reference_dry": sample_dir / "clean_ref.wav",
        "reference_wet": sample_dir / "reference.wav",
        "ground_truth": sample_dir / "target.wav",
    }
    filenames = {
        "source_dry": "source-dry.wav",
        "reference_dry": "reference-dry.wav",
        "reference_wet": "reference-wet.wav",
        "ground_truth": "ground-truth.wav",
    }
    paths: dict[str, str] = {}
    stats: dict[str, dict[str, float]] = {}

    for key, source in sources.items():
        destination = output_dir / filenames[key]
        shutil.copy2(source, destination)
        paths[key] = filenames[key]
        stats[key] = audio_stats(read_audio(destination))
    return paths, stats


def prepare_case(
    fx_chain: Any,
    case: dict[str, Any],
    triplets_dir: Path,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    instrument = str(case["instrument"])
    sample_index = int(case["sample_id"])
    sample_id = f"{sample_index:04d}"
    case_id = f"{instrument}-{sample_id}"
    sample_dir = triplets_dir / instrument / sample_id
    case_output_dir = output_root / case_id
    case_output_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        sample_dir / "clean.wav",
        sample_dir / "clean_ref.wav",
        sample_dir / "reference.wav",
        sample_dir / "target.wav",
        sample_dir / "params.json",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing triplet files: {missing}")

    params_meta = load_json(sample_dir / "params.json")
    shared_paths, shared_stats = copy_shared_audio(sample_dir, case_output_dir)
    arrays = {
        "clean": read_audio(sample_dir / "clean.wav"),
        "clean_ref": read_audio(sample_dir / "clean_ref.wav"),
        "reference": read_audio(sample_dir / "reference.wav"),
        "target": read_audio(sample_dir / "target.wav"),
    }
    record = {
        "id": case_id,
        "instrument": instrument,
        "instrument_label": instrument.title(),
        "sample_id": sample_id,
        "track_source": params_meta.get("track_a", ""),
        "track_reference": params_meta.get("track_b", ""),
        "shared_audio": shared_paths,
        "shared_audio_stats": shared_stats,
        "active_chain": [
            {
                "id": processor,
                "label": PROCESSOR_LABELS[processor],
            }
            for processor in active_processors(params_meta)
        ],
        "key_parameters": format_key_parameters(fx_chain, params_meta),
        "outputs": [],
        "normalization_note": (
            "No per-output loudness normalization was applied. All outputs share "
            "the same source and evaluation target; only Oracle uses reference dry."
        ),
    }
    return record, arrays


def load_relfx_model(checkpoint: Path, device: str) -> tuple[Any, dict[str, Any]]:
    from relfx.checkpoint import load_model

    model, metadata = load_model(checkpoint, device=device)
    model._model_type = "cascaded_v4"
    model.requires_grad_(False)
    model.eval()
    return model, metadata


def load_fxencoderpp_model(
    project_root: Path, checkpoint_path: Path, device: str
) -> tuple[Any, dict[str, Any]]:
    fxencoder_root = project_root / "repos/Fx-Encoder_PlusPlus"
    if not fxencoder_root.is_dir():
        raise NotADirectoryError(fxencoder_root)
    sys.path.insert(0, str(fxencoder_root))
    from fxencoder_plusplus.model import FxEncoderPlusPlus

    model = FxEncoderPlusPlus(
        embed_dim=2048,
        audio_clap_module=False,
        text_clap_module=False,
        extractor_module=False,
        device=device,
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if "epoch" in checkpoint:
        state_dict = checkpoint["state_dict"]
        if next(iter(state_dict)).startswith("module."):
            state_dict = {
                key.removeprefix("module."): value
                for key, value in state_dict.items()
            }
        metadata = {"epoch": checkpoint.get("epoch")}
    else:
        state_dict = checkpoint
        metadata = {"epoch": None}
    model_state = model.state_dict()
    filtered_state = {
        key: value for key, value in state_dict.items() if key in model_state
    }
    model.load_state_dict(filtered_state, strict=False)
    del checkpoint, state_dict, filtered_state
    gc.collect()
    model._eval_sample_rate = SAMPLE_RATE
    model._model_type = "fxencpp"
    model.to(device)
    model.requires_grad_(False)
    model.eval()
    return model, metadata


def state_config(args: argparse.Namespace) -> dict[str, Any]:
    uses_relfx = any(method_id.startswith("relfx_") for method_id in args.methods)
    uses_fxencoderpp = "fxencoderpp" in args.methods
    return {
        "relfx_checkpoint": str(args.relfx_checkpoint.resolve()),
        "relfx_checkpoint_sha256": (
            file_sha256(args.relfx_checkpoint) if uses_relfx else None
        ),
        "fxencoderpp_checkpoint": str(args.fxencoderpp_checkpoint.resolve()),
        "fxencoderpp_checkpoint_sha256": (
            file_sha256(args.fxencoderpp_checkpoint)
            if uses_fxencoderpp
            else None
        ),
        "triplets_dir": str(args.triplets_dir.resolve()),
        "selection": load_json(args.selection),
        "methods": args.methods,
        "n_iters": args.n_iters,
        "num_restarts": args.num_restarts,
        "lr": args.lr,
        "es_patience": args.es_patience,
        "seed": args.seed,
    }


def load_or_create_state(
    args: argparse.Namespace, config: dict[str, Any]
) -> dict[str, Any]:
    if args.state_output.is_file() and not args.force:
        state = load_json(args.state_output)
        if state.get("config") != config:
            raise RuntimeError(
                f"Existing state uses a different configuration: {args.state_output}. "
                "Use --force to start a new run."
            )
        return state
    return {"schema_version": 1, "config": config, "results": {}}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    temporary.replace(path)


def published_ld(
    matcher: Any, output: np.ndarray, target: np.ndarray, device: str
) -> float:
    output_tensor = torch.from_numpy(output).unsqueeze(0).to(device)
    target_tensor = torch.from_numpy(target).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(matcher.mrstft_loss(output_tensor, target_tensor).item())


def optimize_case(
    matcher_class: Any,
    fx_chain: Any,
    model: Any,
    method: dict[str, Any],
    case_record: dict[str, Any],
    arrays: dict[str, np.ndarray],
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    matcher = matcher_class(
        fx_chain,
        sample_rate=SAMPLE_RATE,
        loss_type="embedding",
        lr=args.lr,
        n_iters=args.n_iters,
        num_restarts=args.num_restarts,
        sample_batch_size=1,
        es_patience=args.es_patience,
        device=args.device,
        embed_mode=method["embed_mode"],
    )
    started = time.monotonic()
    result = matcher.match_single(
        clean=arrays["clean"],
        target=arrays["target"],
        reference=arrays["reference"],
        clean_ref=arrays["clean_ref"],
        model=model,
        sample_idx=int(case_record["sample_id"]),
        verbose=args.verbose,
        log_prefix=f"[{case_record['id']}:{method['id']}]",
    )
    elapsed = time.monotonic() - started

    case_output_dir = args.output_dir / case_record["id"]
    filename = f"output-{method['id']}.wav"
    destination = case_output_dir / filename
    write_audio(destination, result["output"])
    published_output = read_audio(destination)
    measured_ld = published_ld(
        matcher, published_output, arrays["target"], args.device
    )
    return {
        "id": method["id"],
        "label": method["label"],
        "kind": method["kind"],
        "description": method["description"],
        "embed_mode": method["embed_mode"],
        "ld": round(measured_ld, 4),
        "optimization_ld": round(float(result["ld"]), 6),
        "audio": filename,
        "audio_stats": audio_stats(published_output),
        "sha256": file_sha256(destination),
        "params": [round(float(value), 8) for value in result["params"]],
        "seed": seed,
        "elapsed_seconds": round(elapsed, 3),
    }


def optimize_case_with_retries(
    matcher_class: Any,
    fx_chain: Any,
    model: Any,
    method: dict[str, Any],
    case_record: dict[str, Any],
    arrays: dict[str, np.ndarray],
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    for attempt in range(MAX_OPTIMIZATION_ATTEMPTS):
        attempt_seed = seed + attempt
        try:
            return optimize_case(
                matcher_class,
                fx_chain,
                model,
                method,
                case_record,
                arrays,
                args,
                attempt_seed,
            )
        except RuntimeError as error:
            no_gradient = "does not require grad" in str(error)
            if not no_gradient or attempt + 1 == MAX_OPTIMIZATION_ATTEMPTS:
                raise
            print(
                f"Retrying {case_record['id']} / {method['id']} after a "
                f"non-finite effect-chain path (seed={attempt_seed})"
            )
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    raise RuntimeError("Optimization retry loop exited unexpectedly")


def validate_paths(args: argparse.Namespace) -> None:
    required_dirs = [args.triplets_dir]
    for path in required_dirs:
        if not path.is_dir():
            raise NotADirectoryError(path)
    if not args.selection.is_file():
        raise FileNotFoundError(args.selection)
    for checkpoint in (args.relfx_checkpoint, args.fxencoderpp_checkpoint):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    if args.n_iters < 1 or args.num_restarts < 1:
        raise ValueError("n-iters and num-restarts must be at least 1")
    if args.lr <= 0 or args.es_patience < 0:
        raise ValueError("lr must be positive and es-patience non-negative")


def main() -> None:
    args = parse_args()
    validate_paths(args)
    selection = load_json(args.selection)
    cases = selection.get("cases", [])
    if args.limit_cases is not None:
        cases = cases[: args.limit_cases]
    if not cases:
        raise ValueError(f"No cases configured in {args.selection}")

    print("Matched website export")
    print(f"  Triplets: {args.triplets_dir}")
    print(f"  RelFx:   {args.relfx_checkpoint}")
    print(f"  Device:  {args.device}")
    print(
        f"  ITO:     {args.n_iters} iterations, {args.num_restarts} restarts, "
        f"lr={args.lr}, patience={args.es_patience}"
    )
    print(f"  Cases:    {len(cases)}")
    for case in cases:
        print(f"    - {case['instrument']}/{int(case['sample_id']):04d}")
    if args.dry_run:
        return

    release_src = args.project_root / "relfx-ismir2026-release/src"
    if not release_src.is_dir():
        raise NotADirectoryError(release_src)
    sys.path.insert(0, str(release_src))
    from relfx.evaluation.ito.fx_chain_wrapper import DifferentiableFxChain
    from relfx.evaluation.ito.parameter_matcher import ParameterMatcher

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fx_chain = DifferentiableFxChain(
        sample_rate=SAMPLE_RATE, device=args.device
    )
    prepared_cases = [
        prepare_case(fx_chain, case, args.triplets_dir, args.output_dir)
        for case in cases
    ]
    config = state_config(args)
    state = load_or_create_state(args, config)
    save_state(args.state_output, state)

    selected_methods = [
        method for method in METHODS if method["id"] in args.methods
    ]
    model_metadata: dict[str, Any] = state.setdefault("model_metadata", {})
    for model_name in ("fxencoderpp", "relfx"):
        model_methods = [
            method for method in selected_methods if method["model"] == model_name
        ]
        pending = [
            (case_record, arrays, method)
            for method in model_methods
            for case_record, arrays in prepared_cases
            if method["id"] in args.force_methods
            or not (
                state["results"]
                .get(case_record["id"], {})
                .get(method["id"])
                and (
                    args.output_dir
                    / case_record["id"]
                    / f"output-{method['id']}.wav"
                ).is_file()
            )
        ]
        if not pending:
            continue

        print(f"Loading {model_name} for {len(pending)} pending runs")
        if model_name == "relfx":
            model, loaded_metadata = load_relfx_model(
                args.relfx_checkpoint, args.device
            )
        else:
            model, loaded_metadata = load_fxencoderpp_model(
                args.project_root,
                args.fxencoderpp_checkpoint,
                args.device,
            )
        model_metadata[model_name] = loaded_metadata
        save_state(args.state_output, state)

        for case_index, (case_record, arrays) in enumerate(prepared_cases):
            for method in model_methods:
                existing = (
                    state["results"]
                    .get(case_record["id"], {})
                    .get(method["id"])
                )
                output_path = (
                    args.output_dir
                    / case_record["id"]
                    / f"output-{method['id']}.wav"
                )
                if (
                    method["id"] not in args.force_methods
                    and existing
                    and output_path.is_file()
                ):
                    print(f"Skipping completed {case_record['id']} / {method['id']}")
                    continue
                seed = args.seed + case_index * 100
                print(f"Optimizing {case_record['id']} / {method['id']} (seed={seed})")
                result = optimize_case_with_retries(
                    ParameterMatcher,
                    fx_chain,
                    model,
                    method,
                    case_record,
                    arrays,
                    args,
                    seed,
                )
                state["results"].setdefault(case_record["id"], {})[
                    method["id"]
                ] = result
                save_state(args.state_output, state)
                print(
                    f"Completed {case_record['id']} / {method['id']}: "
                    f"Ld={result['ld']:.4f}, {result['elapsed_seconds']:.1f}s"
                )

        del model
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    records = []
    for case_record, _ in prepared_cases:
        case_results = state["results"].get(case_record["id"], {})
        case_record["outputs"] = [
            case_results[method["id"]]
            for method in selected_methods
            if method["id"] in case_results
        ]
        records.append(case_record)

    metadata = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "MUSDB18",
        "sample_rate": SAMPLE_RATE,
        "segment_duration_seconds": 10,
        "evaluation": {
            "metric": "Multi-resolution STFT loss",
            "metric_short": "MR-STFT Ld",
            "lower_is_better": True,
            "iterations": args.n_iters,
            "restarts": args.num_restarts,
            "optimizer": "Adam",
            "scheduler": "Cosine annealing",
            "early_stopping_patience": args.es_patience,
            "restart_initialization": "Shared across methods within each case",
            "restart_selection": "Lowest MR-STFT Ld",
            "audio_encoding": "44.1 kHz stereo PCM16 WAV",
            "effect_chain": [PROCESSOR_LABELS[item] for item in PROCESSOR_ORDER],
            "parameter_count": 47,
            "provenance": (
                "Outputs were freshly optimized on fixed MUSDB18 triplets using "
                "the ITO procedure reported in the ISMIR 2026 paper."
            ),
        },
        "models": model_metadata,
        "cases": records,
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata_output.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    print(f"Wrote audio to {args.output_dir}")
    print(f"Wrote metadata to {args.metadata_output}")


if __name__ == "__main__":
    main()
