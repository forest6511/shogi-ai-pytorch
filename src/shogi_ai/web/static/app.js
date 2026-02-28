// Shogi AI Web UI

const ANIMAL_PIECES = {
    0: '\u{1F424}', // CHICK
    1: '\u{1F992}', // GIRAFFE
    2: '\u{1F418}', // ELEPHANT
    3: '\u{1F981}', // LION
    4: '\u{1F414}', // HEN
};

const FULL_PIECES = {
    0: '\u6B69', 1: '\u9999', 2: '\u6842', 3: '\u9280', 4: '\u91D1',
    5: '\u89D2', 6: '\u98DB', 7: '\u7389',
    8: '\u3068', 9: '\u674F', 10: '\u572D', 11: '\u5168',
    12: '\u99AC', 13: '\u9F8D',
};

const ANIMAL_PIECE_VALUES = {
    'CHICK': 0, 'GIRAFFE': 1, 'ELEPHANT': 2, 'LION': 3, 'HEN': 4,
};

const FULL_PIECE_VALUES = {
    'PAWN': 0, 'LANCE': 1, 'KNIGHT': 2, 'SILVER': 3, 'GOLD': 4,
    'BISHOP': 5, 'ROOK': 6, 'KING': 7,
    'PRO_PAWN': 8, 'PRO_LANCE': 9, 'PRO_KNIGHT': 10, 'PRO_SILVER': 11,
    'HORSE': 12, 'DRAGON': 13,
};

const ANIMAL_HAND_NAMES = ['CHICK', 'GIRAFFE', 'ELEPHANT'];
const FULL_HAND_NAMES = ['PAWN', 'LANCE', 'KNIGHT', 'SILVER', 'GOLD', 'BISHOP', 'ROOK'];

let gameId = null;
let gameType = 'animal';
let gameState = null;
let selectedSquare = null;
let selectedHandPiece = null;
let lastAiFrom = -1;
let lastAiTo = -1;

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
    selectedSquare = null;
    selectedHandPiece = null;
    lastAiFrom = -1;
    lastAiTo = -1;
    document.getElementById('move-list').innerHTML = '';
    render();
}

async function makeMove(move) {
    if (!gameId || gameState.is_terminal) return;
    selectedSquare = null;
    selectedHandPiece = null;

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

    // Track AI's last move for highlighting
    if (data.ai_move !== null) {
        const ns = gameState.rows * gameState.cols;
        lastAiFrom = getFromIdx(data.ai_move, ns);
        lastAiTo = getToIdx(data.ai_move, ns);
    } else {
        lastAiFrom = -1;
        lastAiTo = -1;
    }

    const moveList = document.getElementById('move-list');
    moveList.innerHTML += `<div>You: move ${data.player_move}</div>`;
    if (data.ai_move !== null) {
        moveList.innerHTML += `<div>AI: move ${data.ai_move}</div>`;
    }
    moveList.scrollTop = moveList.scrollHeight;

    render();
}

// --- Move decoding helpers ---

function getFromIdx(move, numSquares) {
    if (gameType === 'animal') {
        if (move < 144) return Math.floor(move / 12);
        return -1; // drop
    } else {
        if (move < numSquares * numSquares) return Math.floor(move / numSquares);
        if (move < 2 * numSquares * numSquares)
            return Math.floor((move - numSquares * numSquares) / numSquares);
        return -1; // drop
    }
}

function getToIdx(move, numSquares) {
    if (gameType === 'animal') {
        if (move < 144) return move % 12;
        return (move - 144) % 12;
    } else {
        if (move < numSquares * numSquares) return move % numSquares;
        if (move < 2 * numSquares * numSquares)
            return (move - numSquares * numSquares) % numSquares;
        return (move - 2 * numSquares * numSquares) % numSquares;
    }
}

function isDropMove(move, numSquares) {
    if (gameType === 'animal') return move >= 144;
    return move >= 2 * numSquares * numSquares;
}

function getDropPieceIndex(move, numSquares) {
    if (gameType === 'animal') {
        if (move < 144) return -1;
        return Math.floor((move - 144) / 12);
    } else {
        const dropBase = 2 * numSquares * numSquares;
        if (move < dropBase) return -1;
        return Math.floor((move - dropBase) / numSquares);
    }
}

// --- Rendering ---

function render() {
    if (!gameState) return;

    const pieces = gameType === 'animal' ? ANIMAL_PIECES : FULL_PIECES;
    const boardEl = document.getElementById('board');
    const { rows, cols, squares, hands, legal_moves, current_player, is_terminal, winner } = gameState;
    const numSquares = rows * cols;

    boardEl.className = `board ${gameType}`;
    boardEl.innerHTML = '';

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const idx = r * cols + c;
            const cell = document.createElement('div');
            cell.className = 'cell';

            const sq = squares[idx];
            if (sq) {
                const pieceEl = document.createElement('span');
                const isPromoted = sq.type >= 8; // PRO_PAWN=8..DRAGON=13
                pieceEl.className = `piece ${sq.owner === 1 ? 'gote' : 'sente'}${isPromoted && gameType === 'full' ? ' promoted' : ''}`;
                pieceEl.textContent = pieces[sq.type] || '?';
                cell.appendChild(pieceEl);
            }

            // Highlight AI's last move
            if (lastAiFrom === idx) cell.classList.add('last-move-from');
            if (lastAiTo === idx) cell.classList.add('last-move-to');

            // Highlight selected square
            if (selectedSquare === idx) {
                cell.classList.add('selected');
            }

            // Highlight legal targets for selected board piece
            if (selectedSquare !== null && !is_terminal && current_player === 0) {
                const isTarget = legal_moves.some(m =>
                    getFromIdx(m, numSquares) === selectedSquare &&
                    getToIdx(m, numSquares) === idx
                );
                if (isTarget) {
                    cell.classList.add('legal-target');
                }
            }

            // Highlight legal targets for selected hand piece (drop)
            if (selectedHandPiece !== null && !is_terminal && current_player === 0) {
                const handNames = gameType === 'animal' ? ANIMAL_HAND_NAMES : FULL_HAND_NAMES;
                const ptIndex = handNames.indexOf(selectedHandPiece);
                const isTarget = legal_moves.some(m =>
                    isDropMove(m, numSquares) &&
                    getDropPieceIndex(m, numSquares) === ptIndex &&
                    getToIdx(m, numSquares) === idx
                );
                if (isTarget) {
                    cell.classList.add('legal-target');
                }
            }

            cell.addEventListener('click', () => handleCellClick(idx));
            boardEl.appendChild(cell);
        }
    }

    // Hands with active-turn highlight
    const goteHandEl = document.getElementById('gote-hand');
    const senteHandEl = document.getElementById('sente-hand');
    goteHandEl.className = `hand${current_player === 1 && !is_terminal ? ' active-turn' : ''}`;
    senteHandEl.className = `hand${current_player === 0 && !is_terminal ? ' active-turn' : ''}`;
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
        if (current_player === 0) {
            statusEl.textContent = '\u3042\u306A\u305F\u306E\u756A\u3067\u3059 (\u5148\u624B)';
            statusEl.className = 'status your-turn';
        } else {
            statusEl.textContent = 'AI\u304C\u8003\u3048\u3066\u3044\u307E\u3059...';
            statusEl.className = 'status ai-turn';
        }
    }

    // Hide legal moves buttons (kept in DOM for debugging)
    document.getElementById('legal-moves').style.display = 'none';
}

// --- Click handlers ---

function handleCellClick(idx) {
    if (!gameState || gameState.is_terminal || gameState.current_player !== 0) return;

    const numSquares = gameState.rows * gameState.cols;
    const { squares, legal_moves } = gameState;

    // Board piece is selected
    if (selectedSquare !== null) {
        // Click same square -> deselect
        if (selectedSquare === idx) {
            selectedSquare = null;
            render();
            return;
        }

        // Try to move to this cell
        const matching = legal_moves.filter(m =>
            getFromIdx(m, numSquares) === selectedSquare &&
            getToIdx(m, numSquares) === idx
        );
        if (matching.length === 1) {
            makeMove(matching[0]);
            return;
        } else if (matching.length >= 2) {
            // Promotion choice (lower index = no promote, higher = promote)
            const choice = confirm('\u6210\u308A\u307E\u3059\u304B\uFF1F (OK=\u6210\u308A, Cancel=\u4E0D\u6210)');
            matching.sort((a, b) => a - b);
            makeMove(choice ? matching[matching.length - 1] : matching[0]);
            return;
        }

        // Not a legal target — switch to another own piece if possible
        const sq = squares[idx];
        if (sq && sq.owner === 0) {
            const hasLegalMoves = legal_moves.some(m => getFromIdx(m, numSquares) === idx);
            if (hasLegalMoves) {
                selectedSquare = idx;
                selectedHandPiece = null;
                render();
                return;
            }
        }

        // Deselect
        selectedSquare = null;
        selectedHandPiece = null;
        render();
        return;
    }

    // Hand piece is selected — try to drop on this cell
    if (selectedHandPiece !== null) {
        const handNames = gameType === 'animal' ? ANIMAL_HAND_NAMES : FULL_HAND_NAMES;
        const ptIndex = handNames.indexOf(selectedHandPiece);
        const matching = legal_moves.filter(m =>
            isDropMove(m, numSquares) &&
            getDropPieceIndex(m, numSquares) === ptIndex &&
            getToIdx(m, numSquares) === idx
        );
        if (matching.length === 1) {
            makeMove(matching[0]);
            return;
        }

        // Not a valid drop target — switch to board piece if possible
        const sq = squares[idx];
        if (sq && sq.owner === 0) {
            const hasLegalMoves = legal_moves.some(m => getFromIdx(m, numSquares) === idx);
            if (hasLegalMoves) {
                selectedSquare = idx;
                selectedHandPiece = null;
                render();
                return;
            }
        }

        // Deselect
        selectedSquare = null;
        selectedHandPiece = null;
        render();
        return;
    }

    // Nothing selected — select own piece
    const sq = squares[idx];
    if (sq && sq.owner === 0) {
        const hasLegalMoves = legal_moves.some(m => getFromIdx(m, numSquares) === idx);
        if (hasLegalMoves) {
            selectedSquare = idx;
            selectedHandPiece = null;
            render();
        }
    }
}

function handleHandPieceClick(pieceName) {
    if (!gameState || gameState.is_terminal || gameState.current_player !== 0) return;

    const numSquares = gameState.rows * gameState.cols;
    const handNames = gameType === 'animal' ? ANIMAL_HAND_NAMES : FULL_HAND_NAMES;
    const ptIndex = handNames.indexOf(pieceName);

    const hasDropMoves = gameState.legal_moves.some(m =>
        isDropMove(m, numSquares) && getDropPieceIndex(m, numSquares) === ptIndex
    );

    if (hasDropMoves) {
        if (selectedHandPiece === pieceName) {
            selectedHandPiece = null; // toggle off
        } else {
            selectedHandPiece = pieceName;
        }
        selectedSquare = null;
        render();
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

    const values = gameType === 'animal' ? ANIMAL_PIECE_VALUES : FULL_PIECE_VALUES;

    for (const name of handPieces) {
        const span = document.createElement('span');
        span.className = 'hand-piece';
        if (owner === 0 && selectedHandPiece === name) {
            span.classList.add('selected');
        }
        span.textContent = pieceMap[values[name]] || name;
        if (owner === 0) {
            span.addEventListener('click', (e) => {
                e.stopPropagation();
                handleHandPieceClick(name);
            });
        }
        piecesEl.appendChild(span);
    }
}

// Auto-start
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('new-game-btn').addEventListener('click', newGame);
    newGame();
});
