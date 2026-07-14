from yamloom import (
    Events,
    Job,
    PullRequestEvent,
    PushEvent,
    Workflow,
    WorkflowDispatchEvent,
    action,
    script,
)
from yamloom.actions.ci.coverage import Codecov
from yamloom.actions.github.release import ReleasePlease
from yamloom.actions.github.scm import Checkout
from yamloom.actions.toolchains.rust import InstallRustTool, SetupRust
from yamloom.expressions import context

benchmark_workflow = Workflow(
    name='CodSpeed',
    on=Events(
        push=PushEvent(branches=['main']),
        pull_request=PullRequestEvent(opened=True, synchronize=True, reopened=True),
        workflow_dispatch=WorkflowDispatchEvent(),
    ),
    jobs={
        'benchmarks': Job(
            name='Run Benchmarks',
            runs_on='ubuntu-latest',
            steps=[
                Checkout(),
                SetupRust(),
                InstallRustTool(tool=['cargo-codspeed']),
                script('cargo codspeed build'),
                action(
                    'CodSpeed Action',
                    'CodSpeedHQ/action',
                    ref='v4',
                    with_opts={
                        'mode': 'simulation',
                        'run': 'cargo codspeed run',
                        'token': context.secrets.CODSPEED_TOKEN,
                    },
                ),
            ],
        )
    },
)

coverage_workflow = Workflow(
    name='Coverage',
    on=Events(pull_request=PullRequestEvent(), push=PushEvent()),
    env={'CARGO_TERM_COLOR': 'always'},
    jobs={
        'coverage': Job(
            name='Codecov Coverage Report',
            runs_on='ubuntu-latest',
            steps=[
                Checkout(),
                InstallRustTool(tool=['cargo-llvm-cov']),
                script('cargo llvm-cov --lcov --output-path lcov.info'),
                Codecov(
                    token=context.secrets.CODECOV_TOKEN,
                    files='lcov.info',
                    fail_ci_if_error=True,
                ),
            ],
        )
    },
)

rust_workflow = Workflow(
    name='Rust Checks',
    on=Events(pull_request=PullRequestEvent(), push=PushEvent()),
    env={'CARGO_TERM_COLOR': 'always'},
    jobs={
        'rust-checks': Job(
            runs_on='ubuntu-latest',
            steps=[
                Checkout(),
                SetupRust(components=['clippy']),
                script('cargo clippy', name='Clippy'),
                script('cargo test', name='Test'),
                script('cargo check --all-features', name='Check all features'),
            ],
        ),
    },
)

release_please_workflow = Workflow(
    name='Release Please',
    on=Events(push=PushEvent(branches=['main'])),
    jobs={
        'release-please': Job(
            runs_on='ubuntu-latest',
            steps=[
                Checkout(),
                ReleasePlease(id='release', token=context.secrets.RELEASE_PLEASE),
                script(
                    'cargo publish',
                    condition=ReleasePlease.releases_created(
                        'release'
                    ).from_json_to_bool(),
                    env={'CARGO_REGISTRY_TOKEN': context.secrets.CARGO_REGISTRY_TOKEN},
                ),
            ],
        )
    },
)

if __name__ == '__main__':
    benchmark_workflow.dump('.github/workflows/benchmark.yml')
    coverage_workflow.dump('.github/workflows/coverage.yml')
    rust_workflow.dump('.github/workflows/rust.yml')
    release_please_workflow.dump('.github/workflows/release.yml')
