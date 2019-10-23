#!/bin/bash

set -euxo pipefail

cargo fmt --all -- --check
cargo clippy -- -D warnings
./maturin develop
pytest
