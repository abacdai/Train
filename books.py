from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from engine import State, apply

def square(action):
    return f"{chr(ord('A') + action % 5)}{action // 5 + 1}"

def position_key(state: State):
    return json.dumps(
        state.to_dict(), separators=(",", ":"), sort_keys=True
    )

def short_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

class Book:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                winner INTEGER NOT NULL,
                plies INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS positions (
                state TEXT NOT NULL,
                action INTEGER NOT NULL,
                visits INTEGER NOT NULL,
                wins INTEGER NOT NULL,
                PRIMARY KEY (state, action)
            );

            CREATE TABLE IF NOT EXISTS openings (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                line TEXT NOT NULL,
                visits INTEGER NOT NULL,
                p1_wins INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tactics (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                state TEXT NOT NULL,
                action INTEGER NOT NULL,
                line TEXT NOT NULL,
                kind TEXT NOT NULL,
                examples INTEGER NOT NULL,
                status TEXT NOT NULL
            );
        """)
        self.db.commit()

    def close(self):
        self.db.close()

    def record_game(self, game_id, records, winner):
        """
        records: list dict với state, action, policy.

        game_id bảo đảm resume không ghi cùng ván hai lần.
        """
        if winner not in (-1, 1):
            raise ValueError("Chỉ ghi ván đã có người thắng.")

        with self.db:
            cursor = self.db.execute(
                "INSERT OR IGNORE INTO games VALUES (?, ?, ?)",
                (game_id, winner, len(records)),
            )
            if cursor.rowcount == 0:
                return

            for record in records:
                state = State.from_dict(record["state"])
                key = position_key(state)
                action = int(record["action"])
                won = int(state.turn == winner)

                self.db.execute("""
                    INSERT INTO positions VALUES (?, ?, 1, ?)
                    ON CONFLICT(state, action) DO UPDATE SET
                        visits = visits + 1,
                        wins = wins + excluded.wins
                """, (key, action, won))

            opening = [
                int(record["action"]) for record in records[:8]
            ]
            line_json = json.dumps(opening)
            opening_id = short_id(line_json)
            opening_name = (
                "Khai cuộc "
                + " → ".join(square(x) for x in opening[:2])
                + f" [{opening_id}]"
            )

            self.db.execute("""
                INSERT INTO openings VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(id) DO UPDATE SET
                    visits = visits + 1,
                    p1_wins = p1_wins + excluded.p1_wins
            """, (
                opening_id, opening_name, line_json,
                int(winner == 1),
            ))

            for index, record in enumerate(records):
                state = State.from_dict(record["state"])
                if state.turn != winner:
                    continue

                action = int(record["action"])
                after = apply(state, action)
                before_count = state.owner.count(state.turn)
                after_count = after.owner.count(state.turn)
                lost_cells = before_count - after_count

                if after.winner == state.turn:
                    kind = "Thắng tức thời"
                elif state.started == 3 and lost_cells > 0:
                    kind = "Ứng viên giảm ô rồi thắng"
                else:
                    continue

                key = position_key(state)
                identifier = short_id(f"{key}:{action}:{kind}")
                name = f"{kind} tại {square(action)} [{identifier}]"

                continuation = [
                    int(item["action"])
                    for item in records[index:index + 8]
                ]

                self.db.execute("""
                    INSERT INTO tactics VALUES (
                        ?, ?, ?, ?, ?, ?, 1, ?
                    )
                    ON CONFLICT(id) DO UPDATE SET
                        examples = examples + 1
                """, (
                    identifier,
                    name,
                    key,
                    action,
                    json.dumps(continuation),
                    kind,
                    "observed_self_play_not_independently_verified",
                ))

    def suggest(self, state, minimum_visits=30):
        """
        Chỉ trả gợi ý thống kê ở ĐÚNG trạng thái.
        Không trả về tuyên bố chắc thắng.
        """
        rows = self.db.execute("""
            SELECT action, visits, wins
            FROM positions
            WHERE state = ? AND visits >= ?
            ORDER BY (wins + 1.0) / (visits + 2.0) DESC,
                     visits DESC
        """, (position_key(state), minimum_visits)).fetchall()

        legal = set(state.legal_actions())
        return [
            {
                "action": action,
                "visits": visits,
                "wins": wins,
                "smoothed_win_rate": (wins + 1) / (visits + 2),
            }
            for action, visits, wins in rows
            if action in legal
        ]

    def export(self, output_directory):
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        self.db.row_factory = sqlite3.Row

        data = {}
        for table in ("games", "positions", "openings", "tactics"):
            data[table] = [
                dict(row)
                for row in self.db.execute(f"SELECT * FROM {table}")
            ]

        (output / "books.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with (output / "textbook.md").open(
            "w", encoding="utf-8"
        ) as stream:
            stream.write("# Sách khai cuộc và ứng viên chiến thuật\n\n")
            stream.write(
                "Các kết quả dưới đây là quan sát từ tự đấu, "
                "không phải chứng minh chắc thắng.\n\n"
            )
            stream.write(
                "Thế cờ giữ nguyên hướng bàn; "
                "không gộp theo xoay hoặc đối xứng.\n\n"
            )

            stream.write("## Khai cuộc\n\n")
            stream.write(
                "| Tên | Chuỗi nước | Số ván | P1 thắng |\n"
                "|---|---|---:|---:|\n"
            )
            for row in data["openings"]:
                moves = " → ".join(
                    square(x) for x in json.loads(row["line"])
                )
                stream.write(
                    f"| {row['name']} | {moves} | "
                    f"{row['visits']} | {row['p1_wins']} |\n"
                )

            stream.write("\n## Ứng viên chiến thuật\n\n")
            for row in data["tactics"]:
                stream.write(f"### {row['name']}\n\n")
                state = State.from_dict(json.loads(row["state"]))
                player = "P1 Đỏ" if state.turn == 1 else "P2 Xanh"

                stream.write(f"Người đến lượt: **{player}**.\n\n")
                stream.write("| Hàng | A | B | C | D | E |\n")
                stream.write("|---|---|---|---|---|---|\n")

                for r in range(5):
                    cells = []
                    for c in range(5):
                        i = r * 5 + c
                        owner = state.owner[i]
                        label = "." if owner == 0 else (
                            f"{'Đ' if owner == 1 else 'X'}"
                            f"{state.dots[i]}"
                        )
                        cells.append(label)
                    stream.write(
                        f"| {r + 1} | " + " | ".join(cells) + " |\n"
                    )

                moves = " → ".join(
                    square(x) for x in json.loads(row["line"])
                )
                stream.write(
                    f"\nNước đề xuất: **{square(row['action'])}**.\n\n"
                    f"Biến thể đã quan sát: {moves}.\n\n"
                    f"Số ví dụ thắng đã ghi: {row['examples']}.\n\n"
                    "Trạng thái: chưa kiểm chứng độc lập; "
                    "biến thể không phải phản ứng bắt buộc của đối thủ.\n\n"
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="runs/chain/books.sqlite")
    parser.add_argument("--out", default="runs/chain/textbook")
    args = parser.parse_args()

    book = Book(args.db)
    try:
        book.export(args.out)
    finally:
        book.close()

    print(f"Đã xuất sách vào: {args.out}")
