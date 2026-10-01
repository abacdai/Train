from __future__ import annotations

import os
import random
from dataclasses import dataclass
from functools import cached_property

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from numba import njit
except ImportError:
    njit = None

SIZE = 5
CELLS = SIZE * SIZE

# Một trạng thái không kết thúc có tối đa 25 * 3 = 75 chấm.
# Sau một nước đi, trước khi tiêu hủy ở biên: tối đa 76.
MAX_DOTS = 76

_zrng = random.Random(20260927)
Z_CELLS = [
    [
        [_zrng.getrandbits(64) for _ in range(MAX_DOTS + 1)]
        for _ in range(2)
    ]
    for _ in range(CELLS)
]
Z_TURN = [_zrng.getrandbits(64) for _ in range(2)]
Z_STARTED = [_zrng.getrandbits(64) for _ in range(4)]
Z_WINNER = [_zrng.getrandbits(64) for _ in range(3)]

@dataclass(frozen=True)
class State:
    # owner: 0 = trống, 1 = P1, -1 = P2
    owner: tuple[int, ...] = (0,) * CELLS
    dots: tuple[int, ...] = (0,) * CELLS
    turn: int = 1
    # bit 0: P1 đã khai cuộc; bit 1: P2 đã khai cuộc
    started: int = 0
    winner: int = 0

    def __post_init__(self):
        if len(self.owner) != CELLS or len(self.dots) != CELLS:
            raise ValueError("Bàn cờ phải có đúng 25 ô.")
        if self.turn not in (-1, 1):
            raise ValueError("turn phải là 1 hoặc -1.")
        if self.started not in range(4):
            raise ValueError("started không hợp lệ.")
        if self.winner not in (-1, 0, 1):
            raise ValueError("winner không hợp lệ.")
        for owner, dots in zip(self.owner, self.dots):
            if owner not in (-1, 0, 1):
                raise ValueError("Chủ sở hữu ô không hợp lệ.")
            if not 0 <= dots <= MAX_DOTS:
                raise ValueError("Số chấm nằm ngoài phạm vi hợp lệ.")
            if (owner == 0) != (dots == 0):
                raise ValueError("Ô trống phải có 0 chấm và ngược lại.")
            if self.winner == 0 and dots >= 4:
                raise ValueError("Trạng thái chưa kết thúc phải ổn định.")
        if sum(self.dots) > MAX_DOTS:
            raise ValueError("Tổng số chấm vượt giới hạn trạng thái.")

    @property
    def signature(self):
        # Dùng để đối chiếu, không tin hoàn toàn vào hash 64-bit.
        return (
            self.owner, self.dots, self.turn,
            self.started, self.winner
        )

    @cached_property
    def zobrist(self) -> int:
        h = Z_TURN[0 if self.turn == 1 else 1]
        h ^= Z_STARTED[self.started]
        h ^= Z_WINNER[self.winner + 1]

        for i, (owner, dots) in enumerate(zip(self.owner, self.dots)):
            if owner:
                h ^= Z_CELLS[i][0 if owner == 1 else 1][dots]

        return h

    def legal_actions(self) -> list[int]:
        if self.winner:
            return []

        bit = 1 if self.turn == 1 else 2

        if not self.started & bit:
            return [i for i, owner in enumerate(self.owner) if owner == 0]

        return [
            i for i, owner in enumerate(self.owner)
            if owner == self.turn
        ]

    def mask(self) -> np.ndarray:
        result = np.zeros(CELLS, dtype=np.bool_)
        result[self.legal_actions()] = True
        return result

    def encode(self) -> np.ndarray:
        """
        8 kênh, nhìn từ người sắp đi:
          0..2: ô mình có 1, 2, >=3 chấm
          3..5: ô đối thủ có 1, 2, >=3 chấm
          6: mình đã khai cuộc
          7: đối thủ đã khai cuộc

        Trạng thái terminal có thể còn >=4 chấm.
        MCTS không đưa terminal vào mạng.
        """
        owner = np.asarray(self.owner).reshape(SIZE, SIZE)
        dots = np.minimum(
            np.asarray(self.dots).reshape(SIZE, SIZE), 3
        )
        result = np.zeros((8, SIZE, SIZE), dtype=np.float32)

        for offset, player in ((0, self.turn), (3, -self.turn)):
            for value in (1, 2, 3):
                result[offset + value - 1] = (
                    (owner == player) & (dots == value)
                )

        my_bit = 1 if self.turn == 1 else 2
        enemy_bit = 2 if self.turn == 1 else 1
        result[6].fill(bool(self.started & my_bit))
        result[7].fill(bool(self.started & enemy_bit))
        return result

    def to_dict(self) -> dict:
        return {
            "owner": list(self.owner),
            "dots": list(self.dots),
            "turn": self.turn,
            "started": self.started,
            "winner": self.winner,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "State":
        return cls(
            tuple(int(x) for x in data["owner"]),
            tuple(int(x) for x in data["dots"]),
            int(data["turn"]),
            int(data["started"]),
            int(data["winner"]),
        )

def _resolve_python(owner, dots, action, started, record_events):
    """
    Sửa trực tiếp hai mảng được truyền vào.

    Hàng đợi FIFO; thêm hàng xóm theo Bắc, Nam, Tây, Đông.
    queued ngăn một ô có nhiều bản sao đang chờ trong hàng đợi.

    Event: (chỉ số ô, chủ mới, số chấm mới).
    """
    queue = [action]
    queued = np.zeros(CELLS, dtype=np.bool_)
    queued[action] = True
    cursor = 0

    # Phần tử mồi giúp Numba suy luận kiểu list.
    events = [(-1, 0, 0)]
    winner = 0

    while cursor < len(queue):
        cell = queue[cursor]
        cursor += 1
        queued[cell] = False

        if dots[cell] < 4:
            continue

        player = int(owner[cell])

        owner[cell] = 0
        dots[cell] = 0

        if record_events:
            events.append((cell, 0, 0))

        row, col = cell // SIZE, cell % SIZE
        neighbors = (
            cell - SIZE if row > 0 else -1,
            cell + SIZE if row < SIZE - 1 else -1,
            cell - 1 if col > 0 else -1,
            cell + 1 if col < SIZE - 1 else -1,
        )

        for neighbor in neighbors:
            if neighbor < 0:
                # Hạt vượt biên bị tiêu hủy.
                continue

            owner[neighbor] = player
            dots[neighbor] += 1

            if record_events:
                events.append(
                    (neighbor, player, int(dots[neighbor]))
                )

            if dots[neighbor] >= 4 and not queued[neighbor]:
                queue.append(neighbor)
                queued[neighbor] = True

        # Chỉ kiểm tra sau khi đã truyền đủ bốn hướng.
        if started == 3:
            enemy_exists = False
            for i in range(CELLS):
                if owner[i] == -player:
                    enemy_exists = True
                    break

            if not enemy_exists:
                winner = player
                break

    return winner, events[1:]

_resolve_numba = (
    njit(cache=True)(_resolve_python) if njit is not None else None
)

def apply(
    state: State,
    action: int,
    *,
    trace: bool = False,
    accelerated: bool | None = None,
):
    """
    Trả về State mới.
    Nếu trace=True: trả về (State mới, danh sách event).

    Mặc định dùng engine Python tham chiếu.
    Đặt CHAIN_NUMBA=1 sau khi kiểm thử để bật Numba.
    """
    if state.winner:
        raise ValueError("Ván đấu đã kết thúc.")

    if isinstance(action, (bool, np.bool_)) or not isinstance(
        action, (int, np.integer)
    ):
        raise ValueError("Nước đi phải là chỉ số nguyên từ 0 đến 24.")

    action = int(action)
    if action not in state.legal_actions():
        raise ValueError(f"Nước đi không hợp lệ: {action}")

    owner = np.asarray(state.owner, dtype=np.int16).copy()
    dots = np.asarray(state.dots, dtype=np.int16).copy()

    bit = 1 if state.turn == 1 else 2
    opening = not state.started & bit
    started = state.started | bit

    owner[action] = state.turn
    dots[action] = 3 if opening else dots[action] + 1

    first_event = (
        action, int(owner[action]), int(dots[action])
    )

    if accelerated is None:
        accelerated = os.environ.get("CHAIN_NUMBA", "0") == "1"

    if accelerated and _resolve_numba is None:
        raise RuntimeError(
            "Bạn bật Numba nhưng chưa cài thư viện numba."
        )

    resolver = _resolve_numba if accelerated else _resolve_python
    winner, events = resolver(
        owner, dots, action, started, trace
    )

    result = State(
        tuple(int(x) for x in owner),
        tuple(int(x) for x in dots),
        -state.turn,
        started,
        int(winner),
    )

    if trace:
        return result, [first_event] + list(events)
    return result

class ChainReactionEnv(gym.Env):
    """
    Môi trường hai người chơi luân phiên.

    Reward thuộc góc nhìn NGƯỜI VỪA ĐI:
      +1 nếu nước đó thắng, 0 nếu chưa kết thúc.

    Observation thuộc góc nhìn NGƯỜI SẮP ĐI.
    Không nên nối trực tiếp với thuật toán single-agent
    mà không có adapter cho đối thủ.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode=None):
        super().__init__()
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode chỉ hỗ trợ None hoặc ansi.")
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(CELLS)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(8, SIZE, SIZE),
            dtype=np.float32,
        )
        self.state = State()

    def _info(self):
        return {
            "action_mask": self.state.mask(),
            "to_play": self.state.turn,
            "winner": self.state.winner,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = State()
        return self.state.encode(), self._info()

    def step(self, action):
        actor = self.state.turn
        self.state = apply(self.state, action)
        terminated = self.state.winner != 0
        reward = float(self.state.winner == actor)

        return (
            self.state.encode(),
            reward,
            terminated,
            False,  # Không tự cắt ván theo số lượt.
            self._info(),
        )

    def render(self):
        rows = []
        for row in range(SIZE):
            cells = []
            for col in range(SIZE):
                i = row * SIZE + col
                player = self.state.owner[i]
                prefix = "R" if player == 1 else "B"
                cells.append(
                    "." if player == 0
                    else f"{prefix}{self.state.dots[i]}"
                )
            rows.append(" ".join(f"{cell:>3}" for cell in cells))
        return "\n".join(rows)
