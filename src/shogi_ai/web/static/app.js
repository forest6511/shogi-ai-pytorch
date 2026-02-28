// Shogi AI Web UI

const ANIMAL_PIECES = {
    0: '🐤', // CHICK
    1: '🦒', // GIRAFFE
    2: '🐘', // ELEPHANT
    3: '🦁', // LION
    4: '🐔', // HEN
};

const FULL_PIECES = {
    0: '歩', 1: '香', 2: '桂', 3: '銀', 4: '金',
    5: '角', 6: '飛', 7: '玉',
    8: 'と', 9: '杏', 10: '圭', 11: '全',
    12: '馬', 13: '龍',
};

let gameId = null;
let gameType = 'animal';
let gameState = null;
let selectedMove = null;

async function newGame() {
    gameType = document.getElementById('game-type').value;
    const aiType = document.getElementById('ai-type').value;

    const res = await fetch('/api/new-game', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ game_type: gameType, ai_type: aiType }),
    });
    const data = await res.json();
    gameId = data.game_id;
    gameState = data.state;
    selectedMove = null;
    document.getElementById('move-list').innerHTML = '';
    render();
}

async function makeMove(move) {
    if (!gameId || gameState.is_terminal) return;

    const res = await fetch('/api/move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ game_id: gameId, move: move }),
    });

    if (!res.ok) {
        const err = await res.json();
        alert(err.detail);
        return;
    }

    const data = await res.json();
    gameState = data.state;
    selectedMove = null;

    const moveList = document.getElementById('move-list');
    moveList.innerHTML += `<div>You: move ${data.player_move}</div>`;
    if (data.ai_move !== null) {
        moveList.innerHTML += `<div>AI: move ${data.ai_move}</div>`;
    }
    moveList.scrollTop = moveList.scrollHeight;

    render();
}

function render() {
    if (!gameState) return;

    const pieces = gameType === 'animal' ? ANIMAL_PIECES : FULL_PIECES;
    const boardEl = document.getElementById('board');
    const { rows, cols, squares, hands, legal_moves, current_player, is_terminal, winner } = gameState;

    // Board class
    boardEl.className = `board ${gameType}`;
    boardEl.innerHTML = '';

    // Build move lookup: to_idx -> [move_indices]
    const legalByTarget = {};
    const legalByFrom = {};
    for (const move of legal_moves) {
        // Decode: for simplicity, use the raw index
        // Board moves: from * squares + to (animal) or more complex (full)
        // We'll just show all legal moves as clickable
    }

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const idx = r * cols + c;
            const cell = document.createElement('div');
            cell.className = 'cell';

            const sq = squares[idx];
            if (sq) {
                const pieceEl = document.createElement('span');
                pieceEl.className = `piece ${sq.owner === 1 ? 'gote' : 'sente'}`;
                pieceEl.textContent = pieces[sq.type] || '?';
                cell.appendChild(pieceEl);
            }

            // Highlight legal move targets
            if (selectedMove !== null) {
                for (const m of legal_moves) {
                    const toIdx = getToIdx(m, rows * cols);
                    if (toIdx === idx) {
                        cell.classList.add('legal-target');
                        cell.addEventListener('click', () => makeMove(m));
                    }
                }
            }

            cell.addEventListener('click', () => handleCellClick(idx));
            boardEl.appendChild(cell);
        }
    }

    // Hands
    renderHand('gote-hand', hands[1], pieces, 1);
    renderHand('sente-hand', hands[0], pieces, 0);

    // Status
    const statusEl = document.getElementById('status');
    if (is_terminal) {
        if (winner === 0) {
            statusEl.textContent = 'You win!';
            statusEl.className = 'status win';
        } else if (winner === 1) {
            statusEl.textContent = 'AI wins!';
            statusEl.className = 'status lose';
        } else {
            statusEl.textContent = 'Draw!';
            statusEl.className = 'status';
        }
    } else {
        statusEl.textContent = current_player === 0 ? 'Your turn (先手)' : 'AI thinking...';
        statusEl.className = 'status';
    }

    // Legal moves list (simple)
    const movesEl = document.getElementById('legal-moves');
    movesEl.innerHTML = '';
    if (!is_terminal && current_player === 0) {
        for (const m of legal_moves) {
            const btn = document.createElement('button');
            btn.textContent = `${m}`;
            btn.style.margin = '2px';
            btn.style.fontSize = '11px';
            btn.style.padding = '2px 6px';
            btn.addEventListener('click', () => makeMove(m));
            movesEl.appendChild(btn);
        }
    }
}

function handleCellClick(idx) {
    if (!gameState || gameState.is_terminal) return;
    // Find legal moves that target this cell
    const numSquares = gameState.rows * gameState.cols;
    const matching = gameState.legal_moves.filter(m => {
        return getToIdx(m, numSquares) === idx;
    });
    if (matching.length === 1) {
        makeMove(matching[0]);
    } else if (matching.length > 1) {
        // Multiple moves to same square (e.g., promotion choice)
        const choice = confirm('Promote? (OK=Yes, Cancel=No)');
        // The higher move index is usually the promotion
        matching.sort((a, b) => a - b);
        makeMove(choice ? matching[matching.length - 1] : matching[0]);
    }
}

function getToIdx(move, numSquares) {
    if (gameType === 'animal') {
        if (move < 144) return move % 12;
        return (move - 144) % 12;
    } else {
        if (move < numSquares * numSquares) return move % numSquares;
        if (move < 2 * numSquares * numSquares) return (move - numSquares * numSquares) % numSquares;
        return (move - 2 * numSquares * numSquares) % numSquares;
    }
}

function renderHand(elementId, handPieces, pieceMap, owner) {
    const el = document.getElementById(elementId);
    const piecesEl = el.querySelector('.hand-pieces');
    piecesEl.innerHTML = '';

    if (!handPieces || handPieces.length === 0) {
        piecesEl.textContent = '-';
        return;
    }

    for (const name of handPieces) {
        const span = document.createElement('span');
        span.className = 'hand-piece';
        // Map name to type index
        const typeIdx = Object.entries(pieceMap).find(([k, v]) =>
            v === name || k === name
        );
        span.textContent = name;
        piecesEl.appendChild(span);
    }
}

// Auto-start
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('new-game-btn').addEventListener('click', newGame);
    newGame();
});
