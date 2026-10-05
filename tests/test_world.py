"""World topology: single-world assertions that make a split launch loud.

Pure environment reads, so every case here is a monkeypatched env dict. The
failures being defended are the ones that produce no error at runtime: two
independent worlds each convinced they are the whole job.
"""

import pytest

from iris3b.train.world import assert_single_world, describe_world

_TORCHRUN_ENV = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "NNODES",
)


@pytest.fixture
def env(monkeypatch):
    """A pristine environment; tests set only what the case is about."""
    for name in _TORCHRUN_ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _set(env, **values):
    for name, value in values.items():
        env.setenv(name, str(value))


def test_single_process_defaults(env):
    """No torchrun env at all is a valid one-rank, one-node world."""
    world = describe_world()
    assert (world.rank, world.local_rank) == (0, 0)
    assert (world.world_size, world.local_world_size) == (1, 1)
    assert (world.node_rank, world.num_nodes) == (0, 1)
    assert_single_world(world)


def test_empty_env_values_fall_back_to_defaults(env):
    """An exported-but-empty WORLD_SIZE must not crash the launcher."""
    _set(env, WORLD_SIZE="", LOCAL_WORLD_SIZE="")
    world = describe_world()
    assert (world.world_size, world.local_world_size, world.num_nodes) == (1, 1, 1)


def test_healthy_two_node_world(env):
    """2 nodes x 8 ranks, rank 11 = node 1 local rank 3."""
    _set(
        env,
        RANK=11,
        LOCAL_RANK=3,
        WORLD_SIZE=16,
        LOCAL_WORLD_SIZE=8,
        GROUP_RANK=1,
        GROUP_WORLD_SIZE=2,
        MASTER_ADDR="10.0.1.4",
        MASTER_PORT=29500,
    )
    world = describe_world()
    assert (world.num_nodes, world.node_rank) == (2, 1)
    assert_single_world(world, expected_nodes=2)


def test_num_nodes_derived_when_group_world_size_absent(env):
    """A launcher that only exports WORLD_SIZE/LOCAL_WORLD_SIZE still reads sanely."""
    _set(env, RANK=9, LOCAL_RANK=1, WORLD_SIZE=16, LOCAL_WORLD_SIZE=8, MASTER_ADDR="10.0.1.4")
    world = describe_world()
    assert (world.num_nodes, world.node_rank) == (2, 1)
    assert_single_world(world, expected_nodes=2)


def test_split_world_raises(env):
    """world_size=8 while the job claims 2 nodes: this rank sees half the job."""
    _set(
        env,
        RANK=0,
        LOCAL_RANK=0,
        WORLD_SIZE=8,
        LOCAL_WORLD_SIZE=8,
        GROUP_RANK=0,
        GROUP_WORLD_SIZE=2,
        MASTER_ADDR="10.0.1.4",
    )
    with pytest.raises(ValueError, match="split world"):
        assert_single_world(describe_world())


def test_loopback_master_addr_raises(env):
    """A loopback rendezvous cannot be reached by any other node."""
    _set(
        env,
        RANK=3,
        LOCAL_RANK=3,
        WORLD_SIZE=16,
        LOCAL_WORLD_SIZE=8,
        GROUP_RANK=0,
        GROUP_WORLD_SIZE=2,
        MASTER_ADDR="127.0.0.1",
    )
    with pytest.raises(ValueError, match="loopback"):
        assert_single_world(describe_world())


def test_missing_master_addr_raises_for_multi_node(env):
    """An unset MASTER_ADDR is as unroutable as a loopback one."""
    _set(env, RANK=0, LOCAL_RANK=0, WORLD_SIZE=16, LOCAL_WORLD_SIZE=8, GROUP_WORLD_SIZE=2)
    with pytest.raises(ValueError, match="loopback"):
        assert_single_world(describe_world())


def test_loopback_is_fine_on_one_node(env):
    """--standalone on a single node is the normal case, not a misconfiguration."""
    _set(env, RANK=0, LOCAL_RANK=0, WORLD_SIZE=8, LOCAL_WORLD_SIZE=8, MASTER_ADDR="localhost")
    assert_single_world(describe_world(), expected_nodes=1)


def test_expected_nodes_mismatch_raises(env):
    """The only detectable form of "this world is internally consistent but half the job"."""
    _set(env, RANK=0, LOCAL_RANK=0, WORLD_SIZE=8, LOCAL_WORLD_SIZE=8, MASTER_ADDR="10.0.1.4")
    world = describe_world()
    assert world.num_nodes == 1
    with pytest.raises(ValueError, match="node count mismatch"):
        assert_single_world(world, expected_nodes=2)


def test_nnodes_env_supplies_expected_nodes(env):
    """The launcher exports NNODES, so the check holds even without the flag."""
    _set(env, RANK=0, LOCAL_RANK=0, WORLD_SIZE=8, LOCAL_WORLD_SIZE=8, MASTER_ADDR="10.0.1.4", NNODES=2)
    with pytest.raises(ValueError, match="node count mismatch"):
        assert_single_world(describe_world())


def test_node_local_rank_numbering_raises(env):
    """Ranks 0-7 on node 1 collide with node 0 on data range and checkpoint writer."""
    _set(
        env,
        RANK=0,
        LOCAL_RANK=0,
        WORLD_SIZE=16,
        LOCAL_WORLD_SIZE=8,
        GROUP_RANK=1,
        GROUP_WORLD_SIZE=2,
        MASTER_ADDR="10.0.1.4",
    )
    with pytest.raises(ValueError, match="rank numbering is node-local"):
        assert_single_world(describe_world())
