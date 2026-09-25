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

// Cấu hình bộ não ban đầu
let brain: BrainWeights = {
  tilesWeight: 1.0,
  dotsWeight: 1.0,
  explodingWeight: 2.0,
  totalGamesTrained: 0,
  learnedPatterns: {},
};

// Đọc bộ não cũ nếu đã tồn tại
if (fs.existsSync(BRAIN_FILE)) {
  try {
    const rawData = fs.readFileSync(BRAIN_FILE, 'utf-8');
    brain = JSON.parse(rawData);
    if (!brain.learnedPatterns) brain.learnedPatterns = {};
  } catch (e) {
    console.log('⚠️ Không thể đọc brain.json cũ, tạo bộ não mới...');
  }
}

// 1. Khởi tạo bàn cờ 5x5 trống
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

// 2. Chuẩn hóa bàn cờ về dạng Hash chuỗi để gom nhóm và lọc trùng
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

// 3. LUẬT GAME MỚI: Lượt đầu đặt ô trống (3 chấm), lượt sau chỉ đặt ô của mình (+1 chấm)
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

// 4. MÔ PHỎNG NỔ PHÓNG DẤU CỘNG 4 HƯỚNG
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

// 5. KIỂM TRA THẮNG THUA
function checkWinner(board: CellData[][], movesCount: number): PlayerId | null {
  if (movesCount <= 2) return null;

  let p1Has = hasAnyCell(board, 'p1');
  let p2Has = hasAnyCell(board, 'p2');

  if (p1Has && !p2Has) return 'p1';
  if (p2Has && !p1Has) return 'p2';
  return null;
}

// 6. ĐÁNH GIÁ THẾ CỜ (EVALUATION FUNCTION)
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

// 7. BỘ LỌC VÀ LÀM SẠCH BRAIN (Lọc trùng và loại bỏ thế cờ rác/kém hiệu quả)
function cleanAndFilterBrain() {
  const initialCount = Object.keys(brain.learnedPatterns).length;
  const filteredPatterns: Record<string, PatternMemory> = {};

  for (const [hash, pattern] of Object.entries(brain.learnedPatterns)) {
    // Chỉ giữ lại thế cờ có nước đi mang lại tỉ lệ thắng cao hoặc được kiểm chứng nhiều lần
    if (pattern.bestCounterMove && (pattern.wins >= pattern.losses || pattern.wins >= 3)) {
      filteredPatterns[hash] = pattern;
    }
  }

  brain.learnedPatterns = filteredPatterns;
  const finalCount = Object.keys(brain.learnedPatterns).length;
  console.log(
    `🧹 [BỘ LỌC BRAIN]: Đã tinh lọc bộ nhớ từ ${initialCount.toLocaleString()} -> ${finalCount.toLocaleString()} thế cờ tối ưu.`
  );
}

// 8. HÀM CHẠY HUẤN LUYỆN 1 CHU KỲ
function runTraining(targetGames: number) {
  let p1Wins = 0;

  // Tỷ lệ thử nghiệm nước đi ngẫu nhiên để mở rộng phát hiện các thế cờ mới
  let explorationRate = 0.25;

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

      // TỰ HỌC / TỰ SAI: AI thử ngẫu nhiên để tìm ra cách đánh độc lạ
      if (Math.random() < explorationRate) {
        selectedMove = validMoves[Math.floor(Math.random() * validMoves.length)];
      } else {
        // TỰ HOÀN THIỆN: Lựa chọn nước đi tối ưu nhất theo kinh nghiệm đã học
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
            moveScore += 50.0; // Điểm thưởng lớn cho phản công đỉnh cao
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

        // Cập nhật bộ nhớ sau khi có kết quả
        if (winner) {
          for (const step of gameHistory) {
            if (!brain.learnedPatterns[step.hash]) {
              brain.learnedPatterns[step.hash] = { wins: 0, losses: 0, bestCounterMove: null };
            }

            const patternData = brain.learnedPatterns[step.hash];
            if (step.player === winner) {
              patternData.wins++;
              patternData.bestCounterMove = step.move;
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

    // Giảm dần độ ngẫu nhiên theo thời gian
    if (explorationRate > 0.05) {
      explorationRate *= 0.9999;
    }

    const logInterval = Math.max(1, Math.floor(targetGames / 5));
    if (game % logInterval === 0 || game === targetGames) {
      console.log(
        `  -> Tiến độ: ${game}/${targetGames} trận | P1 Thắng: ${(
          (p1Wins / game) *
          100
        ).toFixed(1)}% | Thế cờ đã ghi nhớ: ${Object.keys(
          brain.learnedPatterns
        ).length.toLocaleString()}`
      );
    }
  }

  // Tự động lọc bớt các thế cờ rác sau khi kết thúc chu kỳ
  cleanAndFilterBrain();

  // Ghi đè bộ não xuống ổ đĩa
  fs.writeFileSync(BRAIN_FILE, JSON.stringify(brain, null, 2));
  console.log(`💾 [LƯU THÀNH CÔNG]: Bộ não đã được ghi vào file ${BRAIN_FILE}`);
}

// 9. VÒNG LẮP HUẤN LUYỆN VĨNH VIỄN
async function startInfiniteTraining() {
  const batchSize = parseInt(process.argv[2], 10) || 10000;
  let cycle = 1;

  console.log(`\n======================================================`);
  console.log(`⚡ KÍCH HOẠT CHẾ ĐỘ TRAIN AI VĨNH VIỄN SIÊU CẤP`);
  console.log(`📦 Quy mô mỗi chu kỳ: ${batchSize.toLocaleString()} trận`);
  console.log(`💡 Bấm Ctrl + C bất kỳ lúc nào để dừng mà KHÔNG mất dữ liệu`);
  console.log(`======================================================\n`);

  while (true) {
    console.log(`🔄 [CHU KỲ ${cycle}] Bắt đầu huấn luyện...`);
    runTraining(batchSize);

    console.log(
      `✅ Hoàn tất chu kỳ ${cycle}. Tổng tích lũy: ${brain.totalGamesTrained.toLocaleString()} trận.`
    );
    console.log(`⏳ Tạm nghỉ 3 giây giải phóng tài nguyên...\n`);

    cycle++;
    await new Promise((resolve) => setTimeout(resolve, 3000));
  }
}

// Kích hoạt chương trình
startInfiniteTraining();
