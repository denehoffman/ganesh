from dataclasses import dataclass

from yamloom import (
    Environment,
    Events,
    Job,
    Matrix,
    Permissions,
    PullRequestEvent,
    PushEvent,
    Strategy,
    Workflow,
    WorkflowDispatchEvent,
    action,
    script,
)
from yamloom.actions.ci.coverage import Codecov
from yamloom.actions.github.artifacts import DownloadArtifact, UploadArtifact
from yamloom.actions.github.release import ReleasePlease
from yamloom.actions.github.scm import Checkout
from yamloom.actions.packaging.python import Maturin
from yamloom.actions.toolchains.python import SetupPython, SetupUV
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
                script('cargo clippy -F python', name='Clippy (python)'),
                script('cargo test', name='Test'),
                script('cargo test -F python', name='Test (python)'),
            ],
        ),
        'check-readme': Job(
            runs_on='ubuntu-latest',
            steps=[
                Checkout(),
                SetupRust(),
                InstallRustTool(tool=['cargo-rdme']),
                script('cargo rdme --check', name='Check README'),
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

PYTHON_VERSIONS = [
    '3.10',
    '3.11',
    '3.12',
    '3.13',
    '3.13t',
    '3.14',
    '3.14t',
    'pypy3.11',
]


@dataclass
class Target:
    runner: str
    target: str
    skip_python_versions: list[str] | None = None

    @property
    def python_versions(self) -> list[str]:
        if self.skip_python_versions is None:
            return PYTHON_VERSIONS
        return [v for v in PYTHON_VERSIONS if v not in self.skip_python_versions]


@dataclass
class TargetJob:
    job_name: str
    short_name: str
    targets: list[Target]

    @property
    def matrix_entry(self) -> list[dict[str, str | list[str]]]:
        entries = []
        for target in self.targets:
            entry = {
                'runner': target.runner,
                'target': target.target,
                'python_versions': target.python_versions,
            }
            if self.short_name == 'windows':
                entry['python_arch'] = (
                    'arm64' if target.target == 'aarch64' else target.target
                )
            entries.append(entry)
        return entries

    def job(self, *, needs: list[str] | None = None, upload: bool = True) -> Job:
        steps = [
            Checkout(),
            script(
                f'printf "%s\n" {context.matrix.platform.python_versions.as_array().join(" ")} >> version.txt'
            ),
            SetupPython(
                python_version_file='version.txt',
                architecture=context.matrix.platform.python_arch.as_str()
                if self.short_name == 'windows'
                else None,
            ),
            Maturin(
                name='Build wheels',
                target=context.matrix.platform.target.as_str(),
                args=f'--release --out dist --interpreter {context.matrix.platform.python_versions.as_array().join(" ")}',
                sccache=~context.github.ref.startswith('ref/tags/'),
                manylinux={'musllinux': 'musllinux_1_2', 'linux': '2_28'}.get(
                    self.short_name
                ),
            ),
        ]
        if upload:
            steps.append(
                UploadArtifact(
                    path='dist',
                    artifact_name=f'{self.short_name}-{context.matrix.platform.target}',
                )
            )
        return Job(
            name=self.job_name,
            strategy=Strategy(fast_fail=False, matrix=Matrix(platform=self.matrix_entry)),
            runs_on=context.matrix.platform.runner.as_str(),
            needs=needs,
            condition=context.github.ref.startswith('refs/tags/')
            | (context.github.event_name == 'workflow_dispatch'),
            steps=steps,
        )


TARGET_JOBS = [
    TargetJob(
        'Build Linux Wheels',
        'linux',
        [
            Target('ubuntu-22.04', target)
            for target in ['x86_64', 'x86', 'aarch64', 'armv7', 's390x', 'ppc64le']
        ],
    ),
    TargetJob(
        'Build (musl) Linux Wheels',
        'musllinux',
        [
            Target('ubuntu-22.04', target)
            for target in ['x86_64', 'x86', 'aarch64', 'armv7']
        ],
    ),
    TargetJob(
        'Build Windows Wheels',
        'windows',
        [
            Target('windows-latest', 'x64'),
            Target('windows-latest', 'x86'),
            Target('windows-11-arm', 'aarch64'),
        ],
    ),
    TargetJob(
        'Build macOS Wheels',
        'macos',
        [
            Target('macos-15-intel', 'x86_64'),
            Target('macos-latest', 'aarch64'),
        ],
    ),
]

test_build_workflow = Workflow(
    name='Build ganesh-rs (Python)',
    on=Events(workflow_dispatch=WorkflowDispatchEvent()),
    jobs={
        **{f'{tj.short_name}': tj.job(upload=False) for tj in TARGET_JOBS},
        'sdist': Job(
            name='Build Source Distribution',
            runs_on='ubuntu-22.04',
            condition=context.github.ref.startswith('refs/tags/')
            | (context.github.event_name == 'workflow_dispatch'),
            steps=[
                Checkout(),
                Maturin(name='Build sdist', command='sdist', args='--out dist'),
                UploadArtifact(path='dist', artifact_name='sdist'),
            ],
        ),
    },
)

python_release_workflow = Workflow(
    name='Build and Release ganesh-rs (Python)',
    on=Events(
        push=PushEvent(branches=['main'], tags=['*']),
        pull_request=PullRequestEvent(opened=True, synchronize=True, reopened=True),
        workflow_dispatch=WorkflowDispatchEvent(),
    ),
    jobs={
        'build-check-test': Job(
            runs_on='ubuntu-latest',
            steps=[
                Checkout(),
                SetupRust(components=['clippy']),
                SetupUV(python_version='3.10'),
                script('cargo clippy', name='Clippy'),
                script('cargo clippy -F python', name='Clippy (python)'),
                script('cargo test', name='Test'),
                script('cargo test -F python', name='Test (python)'),
                script(
                    'uv venv',
                    '. .venv/bin/activate',
                    'echo PATH=$PATH >> $GITHUB_ENV',
                    'uv pip install matplotlib corner joblib polars matplotloom pytest',
                ),
                script('uvx --with "maturin[patchelf]>=1.7,<2" maturin develop --uv'),
                script('uvx ruff check . --extend-exclude=.yamloom.py'),
                script('uvx ty check . --exclude=.yamloom.py'),
                script('uv run pytest'),
            ],
        ),
        **{f'{tj.short_name}': tj.job(needs=['build-check-test']) for tj in TARGET_JOBS},
        'sdist': Job(
            name='Build Source Distribution',
            runs_on='ubuntu-22.04',
            needs=['build-check-test'],
            condition=context.github.ref.startswith('refs/tags/')
            | (context.github.event_name == 'workflow_dispatch'),
            steps=[
                Checkout(),
                Maturin(name='Build sdist', command='sdist', args='--out dist'),
                UploadArtifact(path='dist', artifact_name='sdist'),
            ],
        ),
        'release': Job(
            name='Release',
            runs_on='ubuntu-latest',
            condition=context.github.ref.startswith('refs/tags/')
            | (context.github.event_name == 'workflow_dispatch'),
            needs=[*[f'{tj.short_name}' for tj in TARGET_JOBS], 'sdist'],
            environment=Environment('pypi'),
            steps=[
                Checkout(),
                DownloadArtifact(merge_multiple=True),
                SetupUV(),
                script(
                    'uv publish --trusted-publishing always *.whl *.tar.gz',
                    permissions=Permissions(id_token='write', contents='write'),
                ),
            ],
        ),
    },
)

if __name__ == '__main__':
    benchmark_workflow.dump('.github/workflows/benchmark.yml')
    coverage_workflow.dump('.github/workflows/coverage.yml')
    rust_workflow.dump('.github/workflows/rust.yml')
    release_please_workflow.dump('.github/workflows/release.yml')
    test_build_workflow.dump('.github/workflows/test-build.yml')
    python_release_workflow.dump('.github/workflows/release-python.yml')
