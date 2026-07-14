default: show

run:
    cargo run --release --manifest-path ../../Cargo.toml --example multimodal_ess

show: run
    uv run visualize.py

clean:
    rm -f data.json
