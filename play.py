from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pygame
import torch

from books import Book
from brain import MCTS, PolicyValueNet
from engine import State, apply

CELL = 100
WIDTH = 5 * CELL
HEIGHT = WIDTH + 90

RED = (225, 75, 75)
BLUE = (70, 135, 245)
BACKGROUND = (22, 25, 34)

def load_model(path, device):
    # Chỉ tải checkpoint đáng tin cậy.
    checkpoint = torch.load(
        path, map_location=device, weights_only=False
    )
    config = checkpoint["config"]

    expected_rules = (
        "5x5-3start-fifo-NSWE-immediate-after-explosion-v1"
    )
    if config["rules"] != expected_rules:
        raise ValueError("Checkpoint dùng bộ luật khác.")

    model = PolicyValueNet(
        channels=config["channels"], blocks=config["blocks"]
    ).to(device)
    model.load_state_dict(checkpoint["weights"])
    model.eval()
    return model

def calculate_policy(model, state, simulations, seed):
    rng = np.random.default_rng(seed)
    return MCTS(
        model, simulations=simulations, rng=rng
    ).search(state, noise=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/chain/best.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--simulations", type=int, default=256)
    parser.add_argument("--human", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--book", default=None)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if args.simulations < 1:
        raise ValueError("simulations phải >= 1.")

    model = load_model(args.checkpoint, args.device)
    book = Book(args.book) if args.book else None
    rng = np.random.default_rng(args.seed)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chain Reaction AI")
    font = pygame.font.Font(None, 28)
    small = pygame.font.Font(None, 22)
    clock = pygame.time.Clock()

    state = State()
    display_owner = list(state.owner)
    display_dots = list(state.dots)
    events = deque()

    executor = ThreadPoolExecutor(max_workers=1)
    future = None
    next_event_time = 0
    running = True
    message = ""

    def start_move(action):
        nonlocal state, events, display_owner, display_dots
        old = state
        state, trace = apply(old, action, trace=True)
        display_owner = list(old.owner)
        display_dots = list(old.dots)
        events = deque(trace)

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                    and not args.watch
                    and not state.winner
                    and state.turn == args.human
                    and not events
                    and future is None
                ):
                    x, y = event.pos
                    if 0 <= x < WIDTH and 0 <= y < WIDTH:
                        action = (y // CELL) * 5 + x // CELL
                        if action in state.legal_actions():
                            start_move(action)
                            message = ""

            if not running:
                break

            now = pygame.time.get_ticks()
            if events and now >= next_event_time:
                cell, owner, dots = events.popleft()
                display_owner[cell] = owner
                display_dots[cell] = dots
                next_event_time = now + 70

            ai_turn = args.watch or state.turn != args.human

            if not events and not state.winner and ai_turn:
                if future is None:
                    seed = int(rng.integers(0, 2**31 - 1))
                    future = executor.submit(
                        calculate_policy,
                        model, state, args.simulations, seed,
                    )
                    message = "AI dang tinh..."

                elif future.done():
                    try:
                        policy = future.result()
                    except Exception:
                        # Không tiếp tục với một nước ngẫu nhiên
                        # để che lỗi tìm kiếm.
                        raise

                    future = None
                    action = int(np.argmax(policy))
                    message = "MCTS"

                    if book is not None:
                        hints = book.suggest(state, minimum_visits=30)
                        for hint in hints:
                            proposed = hint["action"]
                            if policy[proposed] >= 0.8 * policy[action]:
                                action = proposed
                                message = (
                                    "Sach thong ke + MCTS "
                                    "(khong bao dam thang)"
                                )
                                break

                    start_move(action)

            screen.fill(BACKGROUND)

            legal = set()
            if not events and not state.winner:
                legal = set(state.legal_actions())

            for cell in range(25):
                row, col = divmod(cell, 5)
                rect = pygame.Rect(
                    col * CELL + 3, row * CELL + 3,
                    CELL - 6, CELL - 6,
                )
                owner = display_owner[cell]
                color = (
                    RED if owner == 1
                    else BLUE if owner == -1
                    else (47, 52, 66)
                )

                pygame.draw.rect(
                    screen, color, rect, border_radius=8
                )
                if cell in legal:
                    pygame.draw.rect(
                        screen, (225, 225, 235), rect,
                        width=2, border_radius=8,
                    )

                label = small.render(
                    f"{chr(65 + col)}{row + 1}",
                    True, (230, 230, 240),
                )
                screen.blit(label, (rect.x + 6, rect.y + 5))

                dots = display_dots[cell]
                cx, cy = rect.center

                offsets = {
                    1: [(0, 0)],
                    2: [(-13, 0), (13, 0)],
                    3: [(0, -13), (-14, 12), (14, 12)],
                }

                if dots in offsets:
                    for dx, dy in offsets[dots]:
                        pygame.draw.circle(
                            screen, (255, 255, 255),
                            (cx + dx, cy + dy), 8,
                        )
                elif dots:
                    text = font.render(
                        str(dots), True, (255, 255, 255)
                    )
                    screen.blit(
                        text, text.get_rect(center=(cx, cy))
                    )

            if events:
                status = "Dang xu ly no..."
            elif state.winner:
                status = (
                    "P1 DO THANG" if state.winner == 1
                    else "P2 XANH THANG"
                )
            else:
                player = "P1 DO" if state.turn == 1 else "P2 XANH"
                status = f"Luot: {player}"

            screen.blit(
                font.render(status, True, (245, 245, 245)),
                (12, WIDTH + 12),
            )
            screen.blit(
                small.render(message, True, (195, 200, 215)),
                (12, WIDTH + 48),
            )

            pygame.display.flip()
            clock.tick(60)

    finally:
        # Nếu đang tìm kiếm, chờ lượt tìm kiếm đó kết thúc.
        executor.shutdown(wait=True, cancel_futures=True)
        if book is not None:
            book.close()
        pygame.quit()

if __name__ == "__main__":
    main()
