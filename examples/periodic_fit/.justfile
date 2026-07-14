default: show

run:
    cargo run --release --manifest-path ../../Cargo.toml --example periodic_fit

show: run
    uv run visualize.py

clean:
    rm -f data.json
