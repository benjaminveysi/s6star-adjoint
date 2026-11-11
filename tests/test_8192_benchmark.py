import sys
from pathlib import Path
from typing import get_type_hints

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.train_ensemble_rdf_mlff import build_parser
import benchmarks.benchmark_8192_adjoint as bench


def test_production_profile_defaults_to_8192_atoms():
    old = sys.argv
    try:
        sys.argv = ["prog"]
        args = build_parser().parse_args()
    finally:
        sys.argv = old
    assert args.num_replicas == 32
    assert args.num_particles == 256
    assert args.num_replicas * args.num_particles == 8192
    assert args.num_steps == 30
    assert args.epochs == 167
    assert args.num_replicas * args.num_steps * args.epochs == 160320


def test_safe_8192_harness_is_pinned():
    assert get_type_hints(bench.Result)["atoms"] is int
