"""Config + discriminated-union registration tests for the mock deployment.

Verifies that ``type: mock`` routes through the ``DeployConfig`` discriminated
union to ``MockDeploymentConfig`` / ``MockDeployment``, the same way every
other deployment (host/local/modal/...) does. This is what makes the mock
selectable from ``agent_config`` with zero changes above ``AgentEnv``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("swerex")

from pydantic import BaseModel, Field  # noqa: E402

from uni_agent.deployment import DeployConfig, MockDeployment, MockDeploymentConfig  # noqa: E402


class _EnvLikeConfig(BaseModel):
    """Mirror of AgentEnvConfig's deployment field: the discriminated union
    only takes effect when it's a *field* of a pydantic model, not as a bare
    Annotated alias."""

    deployment: DeployConfig = Field(default_factory=MockDeploymentConfig)


def test_mock_config_defaults() -> None:
    """The discriminator and tunables have the agreed defaults."""
    cfg = MockDeploymentConfig()
    assert cfg.type == "mock"
    assert cfg.seed is None  # default = random
    assert cfg.observation_scale == 1.0


def test_mock_config_tolerates_docker_only_overrides() -> None:
    """The swebench dataset carries per-sample docker-only fields (image,
    command, container_runtime) in tools_kwargs, which the agent loop deep-
    merges onto the deployment block. MockDeploymentConfig must IGNORE these
    (not reject) so the same dataset drives both local and mock deployments.

    Regression for the ValidationError that killed the first mock perf run:
    ``deployment.mock.image: Extra inputs are not permitted``.
    """
    cfg = MockDeploymentConfig.model_validate(
        {
            "type": "mock",
            "image": "swebench/sweb.eval.x86_64.astropy_1776_astropy-13033",
            "command": "python3 -m swerex.server --auth-token {token}",
            "container_runtime": "docker",
            "extra_run_args": ["-v", "/wheels:/wheels:ro"],
        }
    )
    assert cfg.type == "mock"
    assert cfg.seed is None  # docker-only keys silently dropped


def test_mock_config_accepts_seed_and_scale() -> None:
    cfg = MockDeploymentConfig(seed=42, observation_scale=2.0)
    assert cfg.seed == 42
    assert cfg.observation_scale == 2.0


def test_discriminated_union_routes_type_mock() -> None:
    """A plain dict with ``type: mock`` must parse via the union into the
    MockDeploymentConfig subclass -- this is the wire format coming from YAML
    once it lands in an AgentEnvConfig.deployment field."""
    cfg = _EnvLikeConfig.model_validate({"deployment": {"type": "mock"}})
    assert isinstance(cfg.deployment, MockDeploymentConfig)


def test_get_deployment_returns_mock() -> None:
    """The factory hands back a MockDeployment carrying the seeded MockRuntime."""
    cfg = MockDeploymentConfig(seed=7, observation_scale=1.5)
    dep = cfg.get_deployment(run_id="run-1")
    assert isinstance(dep, MockDeployment)
    assert dep.run_id == "run-1"


@pytest.mark.asyncio
async def test_mock_deployment_exposes_seeded_runtime() -> None:
    """MockDeployment must construct its MockRuntime with seed/scale so the
    route/render behavior is configured before start()."""
    cfg = MockDeploymentConfig(seed=7, observation_scale=1.5)
    dep = cfg.get_deployment(run_id="run-1")
    await dep.start()
    try:
        rt = dep.runtime
        assert rt._seed == 7
        assert rt.observation_scale == 1.5
    finally:
        await dep.stop()
