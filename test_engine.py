import numpy as np
import pytest
import torch

from brain import MCTS, PolicyValueNet
from engine import (
    State, ChainReactionEnv, apply, _resolve_numba
)

def position(p1, p2, turn=1):
    owner = [0] * 25
    dots = [0] * 25

    for player, cells in ((1, p1), (-1, p2)):
        for cell, count in cells.items():
            if owner[cell]:
                raise ValueError("Hai người không thể cùng sở hữu một ô.")
            owner[cell] = player
            dots[cell] = count

    return State(
        tuple(owner), tuple(dots), turn=turn, started=3
    )

def test_each_player_starts_with_three_dots():
    s = apply(State(), 12)
    assert s.owner[12] == 1
    assert s.dots[12] == 3
    assert s.started == 1
    assert s.winner == 0

    with pytest.raises(ValueError):
        apply(s, 12)

    s = apply(s, 24)
    assert s.owner[24] == -1
    assert s.dots[24] == 3
    assert s.started == 3
    assert s.turn == 1
    assert s.winner == 0

def test_only_owned_cells_after_opening():
    s = apply(apply(State(), 0), 24)
    assert s.legal_actions() == [0]

    with pytest.raises(ValueError):
        apply(s, 1)  # Ô trống.

    with pytest.raises(ValueError):
        apply(s, 24)  # Ô đối thủ.

def test_normal_move_adds_exactly_one():
    s = position({12: 1}, {24: 1})
    after = apply(s, 12)
    assert after.dots[12] == 2
    assert after.owner[12] == 1
    assert after.turn == -1

def test_corner_loses_two_particles():
    s = position({0: 3}, {24: 1})
    after = apply(s, 0)

    assert after.owner[0] == 0
    assert after.dots[0] == 0
    assert after.owner[1] == 1
    assert after.owner[5] == 1
    assert after.dots[1] == after.dots[5] == 1
    assert sum(after.dots) == sum(s.dots) + 1 - 2

def test_edge_loses_one_particle():
    s = position({2: 3}, {24: 1})
    after = apply(s, 2)

    for cell in (1, 3, 7):
        assert after.owner[cell] == 1
        assert after.dots[cell] == 1

    assert sum(after.dots) == sum(s.dots) + 1 - 1

def test_center_distributes_four_particles():
    s = position({12: 3}, {24: 1})
    after = apply(s, 12)

    for cell in (7, 17, 11, 13):
        assert after.owner[cell] == 1
        assert after.dots[cell] == 1

    assert sum(after.dots) == sum(s.dots) + 1

def test_capture_preserves_existing_dots_and_adds_one():
    s = position({12: 3}, {7: 2, 24: 1})
    after = apply(s, 12)

    assert after.owner[7] == 1
    assert after.dots[7] == 3
    assert after.winner == 0

def test_chain_reaction_continues_if_enemy_survives():
    s = position({12: 3}, {7: 3, 24: 1})
    after = apply(s, 12)

    assert after.owner[7] == 0
    assert after.dots[7] == 0
    assert after.owner[12] == 1
    assert after.dots[12] == 1

    for cell in (2, 6, 8):
        assert after.owner[cell] == 1
        assert after.dots[cell] == 1

    assert after.winner == 0
    assert max(after.dots) < 4

def test_win_stops_before_processing_pending_explosion():
    s = position({0: 3}, {1: 3})
    after = apply(s, 0)

    assert after.winner == 1
    assert -1 not in after.owner

    # Ô này vừa bị đồng hóa và đã đạt 4.
    # Không nổ tiếp vì ván đã thắng sau lần nổ ô 0.
    assert after.owner[1] == 1
    assert after.dots[1] == 4
    assert after.legal_actions() == []

    with pytest.raises(ValueError):
        apply(after, 1)

def test_fifo_order_is_north_then_south_then_west_then_east():
    s = position(
        {12: 3, 7: 3, 17: 3, 11: 3, 13: 3},
        {24: 1},
    )
    _, events = apply(s, 12, trace=True)

    exploded = [
        cell for cell, owner, dots in events
        if owner == 0 and dots == 0
    ]

    assert exploded[:5] == [12, 7, 17, 11, 13]

def test_trace_reconstructs_result():
    s = position({12: 3}, {7: 3, 24: 1})
    after, events = apply(s, 12, trace=True)

    owner = list(s.owner)
    dots = list(s.dots)

    for cell, player, count in events:
        owner[cell] = player
        dots[cell] = count

    assert tuple(owner) == after.owner
    assert tuple(dots) == after.dots

def test_hash_contains_turn_and_opening_status():
    a = State()
    b = State(turn=-1)
    c = State(started=1)

    assert a.signature != b.signature
    assert a.signature != c.signature
    assert a.zobrist == State().zobrist

def test_gym_contract():
    env = ChainReactionEnv()
    obs, info = env.reset(seed=1)

    assert env.observation_space.contains(obs)
    assert info["action_mask"].sum() == 25

    obs, reward, terminated, truncated, info = env.step(0)
    assert env.observation_space.contains(obs)
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["action_mask"].sum() == 24

def test_random_trajectories_keep_invariants():
    rng = np.random.default_rng(4)

    # Giới hạn này chỉ là ngân sách kiểm thử.
    # Không được dùng làm luật hòa của game.
    for _ in range(10):
        state = State()

        for _ in range(100):
            if state.winner:
                break

            legal = state.legal_actions()
            assert legal
            state = apply(state, int(rng.choice(legal)))

        for owner, dots in zip(state.owner, state.dots):
            assert (owner == 0) == (dots == 0)

        if not state.winner:
            assert max(state.dots) < 4

@pytest.mark.skipif(
    _resolve_numba is None, reason="Chưa cài Numba"
)
def test_numba_matches_reference():
    rng = np.random.default_rng(7)

    for _ in range(5):
        state = State()

        for _ in range(80):
            if state.winner:
                break

            action = int(rng.choice(state.legal_actions()))

            reference, trace_a = apply(
                state, action, trace=True, accelerated=False
            )
            compiled, trace_b = apply(
                state, action, trace=True, accelerated=True
            )

            assert reference == compiled
            assert trace_a == trace_b
            state = reference

def test_mcts_masks_illegal_moves():
    model = PolicyValueNet(channels=8, blocks=1)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)

    state = apply(apply(State(), 0), 24)
    policy = MCTS(
        model, simulations=4,
        rng=np.random.default_rng(1),
    ).search(state)

    assert np.isclose(policy.sum(), 1.0)
    assert policy[0] == 1.0
    assert np.count_nonzero(policy) == 1

def test_mcts_finds_immediate_win():
    model = PolicyValueNet(channels=8, blocks=1)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)

    state = position({0: 3, 20: 1}, {1: 3})
    policy = MCTS(
        model, simulations=32,
        rng=np.random.default_rng(2),
    ).search(state)

    assert int(np.argmax(policy)) == 0
