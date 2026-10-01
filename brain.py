from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from engine import CELLS, State, apply

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.conv1 = nn.Conv2d(
            channels, channels, 3, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(
            channels, channels, 3, padding=1, bias=False
        )
        self.norm2 = nn.GroupNorm(groups, channels)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x + y)

class PolicyValueNet(nn.Module):
    def __init__(self, channels=64, blocks=4):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1

        self.stem = nn.Sequential(
            nn.Conv2d(8, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.ReLU(),
        )
        self.body = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(blocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 8, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 25, CELLS),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 4, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 25, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        h = self.body(self.stem(x))
        return self.policy_head(h), self.value_head(h).squeeze(-1)

@torch.inference_mode()
def evaluate(model, state: State):
    if state.winner:
        raise ValueError("Không gọi mạng cho trạng thái terminal.")

    model.eval()
    device = next(model.parameters()).device
    x = torch.from_numpy(state.encode()).unsqueeze(0).to(device)
    logits, value = model(x)

    mask = torch.from_numpy(state.mask()).to(device)
    logits = logits[0].masked_fill(~mask, -1e9)
    policy = torch.softmax(logits, dim=0).cpu().numpy()

    return policy.astype(np.float64), float(value.item())

class Node:
    def __init__(self, state: State):
        self.state = state
        self.expanded = False
        self.p = np.zeros(CELLS, dtype=np.float64)
        self.n = np.zeros(CELLS, dtype=np.int64)
        self.w = np.zeros(CELLS, dtype=np.float64)
        self.children = {}

class MCTS:
    def __init__(
        self,
        model,
        simulations=128,
        cpuct=1.7,
        max_depth=256,
        rng=None,
    ):
        if simulations < 1 or max_depth < 1:
            raise ValueError("simulations và max_depth phải >= 1.")

        self.model = model
        self.simulations = simulations
        self.cpuct = cpuct
        self.max_depth = max_depth
        self.rng = rng if rng is not None else np.random.default_rng()

        # hash -> list Node: bucket cho trường hợp va chạm.
        self.table = {}

    def _node(self, state):
        bucket = self.table.setdefault(state.zobrist, [])
        for node in bucket:
            if node.state.signature == state.signature:
                return node

        node = Node(state)
        bucket.append(node)
        return node

    def _expand(self, node):
        policy, value = evaluate(self.model, node.state)
        node.p = policy
        node.expanded = True
        return value

    def search(self, state, *, noise=False):
        if state.winner:
            raise ValueError("Không tìm kiếm trên ván đã kết thúc.")

        # Giới hạn bộ nhớ trong phạm vi một lần tìm kiếm.
        self.table = {}
        root = self._node(state)
        self._expand(root)

        root_prior = root.p.copy()
        legal = state.legal_actions()

        if noise:
            random_prior = self.rng.dirichlet(
                np.full(len(legal), 0.3)
            )
            root_prior[legal] = (
                0.75 * root_prior[legal] + 0.25 * random_prior
            )

        for _ in range(self.simulations):
            node = root
            path = []
            seen = set()
            depth = 0

            while True:
                s = node.state

                if s.winner:
                    value = 1.0 if s.winner == s.turn else -1.0
                    break

                if s.signature in seen:
                    # Chỉ ngắt nhánh mô phỏng lặp.
                    # Đây là heuristic tìm kiếm, KHÔNG phải kết quả hòa.
                    value = 0.0
                    break

                seen.add(s.signature)

                if depth >= self.max_depth:
                    # Ước lượng lá; không cắt ván thật.
                    _, value = evaluate(self.model, s)
                    break

                if not node.expanded:
                    value = self._expand(node)
                    break

                actions = s.legal_actions()
                if not actions:
                    raise RuntimeError(
                        "Trạng thái chưa kết thúc nhưng không có nước đi."
                    )

                prior = root_prior if node is root else node.p
                total = max(1, int(node.n.sum()))

                q = np.divide(
                    node.w,
                    node.n,
                    out=np.zeros(CELLS, dtype=np.float64),
                    where=node.n > 0,
                )
                u = (
                    self.cpuct * prior * math.sqrt(total)
                    / (1.0 + node.n)
                )

                scores = q[actions] + u[actions]
                # Phá hòa có seed, tránh thiên lệch chỉ số ô.
                scores = scores + self.rng.random(len(actions)) * 1e-10
                action = actions[int(np.argmax(scores))]

                path.append((node, action))

                if action not in node.children:
                    child_state = apply(s, action)
                    node.children[action] = self._node(child_state)

                node = node.children[action]
                depth += 1

            # Value tại lá thuộc người sắp đi ở lá.
            for parent, action in reversed(path):
                value = -value
                parent.n[action] += 1
                parent.w[action] += value

        visits = root.n.astype(np.float64)
        policy = visits / visits.sum()
        return policy.astype(np.float32)

def choose_action(policy, rng, temperature=1.0):
    policy = np.asarray(policy, dtype=np.float64)

    if temperature <= 1e-8:
        return int(np.argmax(policy))

    positive = policy > 0
    if not positive.any():
        raise ValueError("Policy không có nước đi được thăm.")

    log_weights = np.full_like(policy, -np.inf)
    log_weights[positive] = np.log(policy[positive]) / temperature
    log_weights[positive] -= log_weights[positive].max()

    probabilities = np.zeros_like(policy)
    probabilities[positive] = np.exp(log_weights[positive])
    probabilities /= probabilities.sum()

    return int(rng.choice(len(policy), p=probabilities))

def train_network(
    model,
    replay,
    rng,
    *,
    steps=200,
    batch_size=128,
    learning_rate=2e-3,
):
    """
    Replay item: (observation, legal_mask, target_policy, outcome).

    Ưu tiên các mẫu có value-error lớn từ những lần học trước.
    Giữ 50% xác suất lấy mẫu đều để không bỏ quên dữ liệu khác.
    Dùng importance weights để giảm bias do ưu tiên mẫu.
    """
    if not replay:
        raise ValueError("Replay buffer đang trống.")
    if steps < 1 or batch_size < 1:
        raise ValueError("steps và batch_size phải >= 1.")

    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )

    model.train()
    priorities = np.ones(len(replay), dtype=np.float64)

    sums = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
    }

    for _ in range(steps):
        weighted = priorities ** 0.6
        probabilities = (
            0.5 / len(replay)
            + 0.5 * weighted / weighted.sum()
        )
        indices = rng.choice(
            len(replay), size=batch_size, replace=True,
            p=probabilities,
        )
        items = [replay[int(i)] for i in indices]

        x = torch.from_numpy(
            np.stack([item[0] for item in items])
        ).to(device)

        mask = torch.from_numpy(
            np.stack([item[1] for item in items])
        ).to(device)

        target_policy = torch.from_numpy(
            np.stack([item[2] for item in items])
        ).to(device)

        outcome = torch.tensor(
            [item[3] for item in items],
            dtype=torch.float32, device=device,
        )

        weights = (len(replay) * probabilities[indices]) ** -1.0
        weights = weights / weights.max()
        weights = torch.tensor(
            weights, dtype=torch.float32, device=device
        )

        logits, predicted_value = model(x)
        logits = logits.masked_fill(~mask, -1e9)

        policy_losses = -(
            target_policy * F.log_softmax(logits, dim=1)
        ).sum(dim=1)
        value_losses = (predicted_value - outcome).square()

        policy_loss = (weights * policy_losses).mean()
        value_loss = (weights * value_losses).mean()
        loss = policy_loss + value_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        errors = (
            (predicted_value.detach() - outcome).abs()
            .cpu().numpy()
        )
        priorities[indices] = errors + 0.05

        sums["loss"] += float(loss.item())
        sums["policy_loss"] += float(policy_loss.item())
        sums["value_loss"] += float(value_loss.item())

    model.eval()
    return {key: value / steps for key, value in sums.items()}
