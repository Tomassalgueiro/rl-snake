"""Estado + máscara de segurança partilhados entre o treino (headless) e o watch (pygame).

Funciona com qualquer jogo que tenha .body, .direction e .food, por isso
SnakeGameHeadless e SnakeGameAI produzem exatamente o mesmo estado.
"""
from collections import deque
import numpy as np

from game_headless import GRID_SIZE, ARENA_LEFT, ARENA_TOP, ARENA_WIDTH, ARENA_HEIGHT

COLS = ARENA_WIDTH // GRID_SIZE
ROWS = ARENA_HEIGHT // GRID_SIZE
N_CELLS = COLS * ROWS
STATE_SIZE = 23

# Direções em sentido horário (por nome, porque game.py e game_headless.py têm enums diferentes)
CLOCKWISE = ['RIGHT', 'DOWN', 'LEFT', 'UP']
DELTAS = {'RIGHT': (1, 0), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'UP': (0, -1)}
TURNS = (0, 1, -1)  # mesma ordem das ações: [frente, direita, esquerda]

# Vizinhos pré-calculados de cada célula (índice plano r * COLS + c)
NEIGHBORS = []
for _i in range(N_CELLS):
    _c, _r = _i % COLS, _i // COLS
    _nb = []
    if _c > 0:
        _nb.append(_i - 1)
    if _c < COLS - 1:
        _nb.append(_i + 1)
    if _r > 0:
        _nb.append(_i - COLS)
    if _r < ROWS - 1:
        _nb.append(_i + COLS)
    NEIGHBORS.append(tuple(_nb))


def to_cell(pt):
    return (pt.x - ARENA_LEFT) // GRID_SIZE, (pt.y - ARENA_TOP) // GRID_SIZE


def to_index(pt):
    c, r = to_cell(pt)
    return r * COLS + c


def food_distance(game):
    hc, hr = to_cell(game.body[0])
    fc, fr = to_cell(game.food)
    return abs(hc - fc) + abs(hr - fr)


def get_state(game):
    """Devolve (estado, máscara). A máscara marca as ações seguras: sem colisão
    e com a cauda ainda alcançável depois do movimento (seguir a cauda é sempre
    uma saída). Se nenhuma for segura, fica só a ação com mais espaço livre."""
    body = game.body
    length = len(body)
    head_c, head_r = to_cell(body[0])

    # Obstáculos depois do movimento: o corpo sem a cauda (a cauda sai do sítio).
    # Se a cobra comer, a cauda fica, mas continua a ser o alvo de "alcançar a cauda".
    blocked = bytearray(N_CELLS)
    for pt in body[:-1]:
        blocked[to_index(pt)] = 1
    tail_idx = to_index(body[-1])
    free_cells = N_CELLS - (length - 1)

    # Componentes ligadas de células livres, calculadas só quando necessárias
    labels = [0] * N_CELLS
    sizes = [0]

    def label_of(start):
        if labels[start]:
            return labels[start]
        lab = len(sizes)
        labels[start] = lab
        queue = deque([start])
        count = 0
        while queue:
            cur = queue.popleft()
            count += 1
            for nxt in NEIGHBORS[cur]:
                if not blocked[nxt] and not labels[nxt]:
                    labels[nxt] = lab
                    queue.append(nxt)
        sizes.append(count)
        return lab

    tail_label = label_of(tail_idx)
    dir_idx = CLOCKWISE.index(game.direction.name)

    danger, area, reach, fit = [], [], [], []
    for turn in TURNS:
        dc, dr = DELTAS[CLOCKWISE[(dir_idx + turn) % 4]]
        c, r = head_c + dc, head_r + dr
        if not (0 <= c < COLS and 0 <= r < ROWS) or blocked[r * COLS + c]:
            danger.append(1.0)
            area.append(0.0)
            reach.append(0.0)
            fit.append(0.0)
            continue
        lab = label_of(r * COLS + c)
        danger.append(0.0)
        area.append(sizes[lab] / free_cells)
        reach.append(float(lab == tail_label))
        fit.append(min(sizes[lab] / length, 1.0))  # o corpo cabe nesta zona?

    direction = game.direction.name
    food_c, food_r = to_cell(game.food)
    tail_c, tail_r = to_cell(body[-1])
    span = COLS + ROWS

    state = [
        # 1. Perigos imediatos (3)
        *danger,
        # 2. Direção (4)
        direction == 'LEFT', direction == 'RIGHT', direction == 'UP', direction == 'DOWN',
        # 3. Comida relativa (4)
        food_c < head_c, food_c > head_c, food_r < head_r, food_r > head_r,
        # 4. Área alcançável por ação, normalizada pelas células livres (3)
        *area,
        # 5. Cauda alcançável por ação (3)
        *reach,
        # 6. A zona tem espaço para o corpo inteiro? (3)
        *fit,
        # 7. Distâncias e tamanho (3)
        (abs(head_c - tail_c) + abs(head_r - tail_r)) / span,
        (abs(head_c - food_c) + abs(head_r - food_r)) / span,
        length / N_CELLS,
    ]

    safe = [d == 0.0 and t == 1.0 for d, t in zip(danger, reach)]
    if any(safe):
        mask = safe
    elif min(danger) == 0.0:
        best = max(a for a, d in zip(area, danger) if d == 0.0)
        mask = [d == 0.0 and a == best for a, d in zip(area, danger)]
    else:
        mask = [True, True, True]  # morte inevitável

    return np.array(state, dtype=np.float32), np.array(mask, dtype=bool)


def terminal_state():
    """Estado de substituição para transições finais (não é usado no bootstrap)."""
    return np.zeros(STATE_SIZE, dtype=np.float32), np.ones(3, dtype=bool)
