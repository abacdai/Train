import fs from 'fs';

type PlayerId = 'p1' | 'p2';

interface CellData {
  playerId: PlayerId | null;
  dots: number;
}

interface PatternMemory {
  wins: number;
  losses: number;
  bestCounterMove: { row: number; col: number } | null;
}

interface BrainWeights {
  tilesWeight: number;
  dotsWeight: number;
  explodingWeight: number;
  totalGamesTrained: number;
  learnedPatterns: Record<string, PatternMemory>;
}

const BOARD_SIZE = 5;
const BRAIN_FILE = './brain.json';

let brain: BrainWeights = {
  tilesWeight: 1.0,
  dotsWeight: 1.0,
  explodingWeight: 2.0,
  totalGamesTrained: 0,
  learnedPatterns: {},
};

if (fs.existsSync(BRAIN_FILE)) {
  try {
    const rawData = fs.readFileSync(BRAIN_FILE, 'utf-8');
    brain = JSON.parse(rawData);
    if (!brain.learnedPatterns) brain.learnedPatterns = {};
  } catch (e) {
    console.log('⚠️ Không thể đọc brain.json cũ, tạo bộ não mới...');
  }
}

function createEmptyBoard(): CellData[][] {
  const board: CellData[][] = [];
  for (let r = 0; r < BOARD_SIZE; r++) {
    const row: CellData[] = [];
    for (let c = 0; c < BOARD_SIZE; c++) {
      row.push({ playerId: null, dots: 0 });
    }
    board.push(row);
  }
  return board;
}

function cloneBoard(board: CellData[][]): CellData[][] {
  return board.map((row) => row.map((cell) => ({ ...cell })));
}

// Chuẩn hóa bàn cờ về dạng Hash để lọc trùng
function getBoardHash(board: CellData[][]): string {
  return board
    .map((row) =>
      row
        .map((cell) => (cell.playerId ? `${cell.playerId[1]}${cell.dots}` : '0'))
        .join('')
    )
    .join('-');
}

function hasAnyCell(board: CellData[][], playerId: PlayerId): boolean {
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      if (board[r][c].playerId === playerId) return true;
    }
  }
  return false;
}

// LUẬT GAME CHUẨN: Lượt đầu đi ô trống (3 chấm), lượt sau CHỈ ĐƯỢC đi ô của mình (+1 chấm)
function getValidMoves(board: CellData[][], playerId: PlayerId): { row: number; col: number }[] {
  const moves: { row: number; col: number }[] = [];
  const isFirstPlacement = !hasAnyCell(board, playerId);

  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      const cell = board[r][c];
      if (isFirstPlacement) {
        if (cell.playerId === null) moves.push({ row: r, col: c });
      } else {
        if (cell.playerId === playerId) moves.push({ row: r, col: c });
      }
    }
  }
  return moves;
}

// LOGIC NỔ PHÓNG DẤU CỘNG 4 HƯỚNG
function applyMoveAndExplode(
  board: CellData[][],
  row: number,
  col: number,
  playerId: PlayerId
): CellData[][] {
  const simBoard = cloneBoard(board);
  const cell = simBoard[row][col];

  if (cell.playerId === null) {
    simBoard[row][col] = { playerId, dots: 3 };
  } else {
    simBoard[row][col].dots += 1;
  }

  let exploding = true;
  let chainCount = 0;

  while (exploding && chainCount < 100) {
    exploding = false;
    chainCount++;
    const toExplode: { r: number; c: number; owner: PlayerId }[] = [];

    for (let r = 0; r < BOARD_SIZE; r++) {
      for (let c = 0; c < BOARD_SIZE; c++) {
        if (simBoard[r][c].dots >= 4 && simBoard[r][c].playerId !== null) {
          toExplode.push({ r, c, owner: simBoard[r][c].playerId! });
        }
      }
    }

    if (toExplode.length > 0) {
      exploding = true;
      for (const exp of toExplode) {
        const { r, c, owner } = exp;

        // Ô nổ trở thành ô trống
        simBoard[r][c] = { playerId: null, dots: 0 };

        // Phóng dấu cộng 4 hướng
        const neighbors: [number, number][] = [
          [r - 1, c],
          [r + 1, c],
          [r, c - 1],
          [r, c + 1],
        ];

        for (const [nr, nc] of neighbors) {
          if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
            if (simBoard[nr][nc].dots > 0) {
              simBoard[nr][nc].playerId = owner;
              simBoard[nr][nc].dots += 1;
            }
          }
        }
      }
    }
  }

  return simBoard;
}

function checkWinner(board: CellData[][], movesCount: number): PlayerId | null {
  if (movesCount <= 2) return null;

  let p1Has = hasAnyCell(board, 'p1');
  let p2Has = hasAnyCell(board, 'p2');

  if (p1Has && !p2Has) return 'p1';
  if (p2Has && !p1Has) return 'p2';
  return null;
}

function evaluateBoard(board: CellData[][], playerId: PlayerId): number {
  let score = 0;
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      const cell = board[r][c];
      if (cell.playerId === playerId) {
        score += brain.tilesWeight;
        score += cell.dots * brain.dotsWeight;
        if (cell.dots === 3) score += brain.explodingWeight;
      }
    }
  }
  return score;
}

// BỘ LỌC VÀ TỐI ƯU BRAIN (Tự động lọc trùng và loại bỏ các thế cờ kém hiệu quả)
function cleanAndFilterBrain() {
  const initialCount = Object.keys(brain.learnedPatterns).length;
  const filteredPatterns: Record<string, PatternMemory> = {};

  for (const [hash, pattern] of Object.entries(brain.learnedPatterns)) {
    // Chỉ giữ lại những thế cờ có nước đi mang lại tỷ lệ thắng dương hoặc được thử nghiệm đủ tốt
    if (pattern.bestCounterMove && (pattern.wins >= pattern.losses || pattern.wins >= 3)) {
      filteredPatterns[hash] = pattern;
    }
  }

  brain.learnedPatterns = filteredPatterns;
  const finalCount = Object.keys(brain.learnedPatterns).length;
  console.log(`🧹 [BỘ LỌC BRAIN]: Đã tinh lọc bộ nhớ từ ${initialCount.toLocaleString()} -> ${finalCount.toLocaleString()} thế cờ tối ưu.`);
}

function runTraining(targetGames: number) {
  console.log(`🚀 Kích hoạt chế độ AI Tự Học Siêu Cấp với ${targetGames.toLocaleString()} trận...`);
  let p1Wins = 0;

  // Tỷ lệ thử nghiệm nước đi ngẫu nhiên để khám phá (Exploration Rate)
  let explorationRate = 0.3;

  for (let game = 1; game <= targetGames; game++) {
    let board = createEmptyBoard();
    let currentPid: PlayerId = 'p1';
    let movesCount = 0;

    const gameHistory: { hash: string; move: { row: number; col: number }; player: PlayerId }[] = [];

    while (true) {
      movesCount++;
      const validMoves = getValidMoves(board, currentPid);

      if (validMoves.length === 0) break;

      const currentHash = getBoardHash(board);
      let selectedMove = validMoves[0];

      // THỬ NGHIỆM: Đôi khi AI chọn đi ngẫu nhiên để thử các phương án mới
      if (Math.random() < explorationRate) {
        selectedMove = validMoves[Math.floor(Math.random() * validMoves.length)];
      } else {
        // TỐI ƯU: AI chọn nước đi dựa trên kinh nghiệm từ bộ não
        let bestScore = -Infinity;
        const knownPattern = brain.learnedPatterns[currentHash];

        for (const move of validMoves) {
          const nextBoard = applyMoveAndExplode(board, move.row, move.col, currentPid);
          let moveScore = evaluateBoard(nextBoard, currentPid);

          if (
            knownPattern &&
            knownPattern.bestCounterMove &&
            knownPattern.bestCounterMove.row === move.row &&
            knownPattern.bestCounterMove.col === move.col
          ) {
            moveScore += 50.0; // Điểm thưởng lớn cho thế cờ đã từng chứng minh thắng lợi
          }

          if (moveScore > bestScore) {
            bestScore = moveScore;
            selectedMove = move;
          }
        }
      }

      gameHistory.push({ hash: currentHash, move: selectedMove, player: currentPid });
      board = applyMoveAndExplode(board, selectedMove.row, selectedMove.col, currentPid);

      const winner = checkWinner(board, movesCount);

      if (winner || movesCount > 250) {
        if (winner === 'p1') p1Wins++;

        // HỌC TẬP VÀ RÚT KINH NGHIỆM: Cập nhật kết quả trận đấu cho cả 2 bên
        if (winner) {
          for (const step of gameHistory) {
            if (!brain.learnedPatterns[step.hash]) {
              brain.learnedPatterns[step.hash] = { wins: 0, losses: 0, bestCounterMove: null };
            }

            const patternData = brain.learnedPatterns[step.hash];
            if (step.player === winner) {
              patternData.wins++;
              patternData.bestCounterMove = step.move; // Ghi nhớ nước đi chiến thắng
            } else {
              patternData.losses++;
            }
          }
        }
        break;
      }

      currentPid = currentPid === 'p1' ? 'p2' : 'p1';
    }

    brain.totalGamesTrained++;

    // Giảm dần tỷ lệ thử nghiệm ngẫu nhiên theo thời gian
    if (explorationRate > 0.05) {
      explorationRate *= 0.9999;
    }

    // In tiến độ và chạy lọc dữ liệu định kỳ
    const logInterval = Math.max(1, Math.floor(targetGames / 10));
    if (game % logInterval === 0 || game === targetGames) {
      console.log(
        `[Trận ${game}/${targetGames}] | P1 Thắng: ${((p1Wins / game) * 100).toFixed(1)}% | Thế cờ đã lưu: ${Object.keys(brain.learnedPatterns).length.toLocaleString()}`
      );
    }
  }

  // Chạy bộ lọc loại bỏ các thế cờ trùng/kém trước khi lưu
  cleanAndFilterBrain();

  fs.writeFileSync(BRAIN_FILE, JSON.stringify(brain, null, 2));
  console.log(`\n💾 Đã lưu bộ não nâng cấp vào file ${BRAIN_FILE}`);
}

const inputGames = parseInt(process.argv[2], 10) || 50000;
runTraining(inputGames);
