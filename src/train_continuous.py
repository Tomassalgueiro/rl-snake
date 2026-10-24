import os
import random
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game_headless import SnakeGameHeadless
from features import STATE_SIZE, get_state, terminal_state, food_distance

LATEST_PATH = "model/dqn_latest.pth"
BEST_PATH = "model/dqn_best.pth"

NUM_ENVS = 8                # Jogos em simultâneo (inferência em batch)
MAX_MEMORY = 200_000
BATCH_SIZE = 256
LR = 0.00025
GAMMA = 0.99
TRAIN_EVERY = 4             # Passos de ambiente por cada passo de gradiente
WARMUP = 10_000             # Transições na memória antes de começar a treinar
TARGET_SYNC = 2_000         # Passos de gradiente entre cópias para a Target Network
EPS_START, EPS_END, EPS_DECAY_STEPS = 1.0, 0.02, 300_000
SHAPING = 0.1               # Peso do reward shaping por distância à comida

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. Rede Neural
class Advanced_DQN(nn.Module):
    def __init__(self, input_size=STATE_SIZE, hidden_size=256, output_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)


def masked_argmax(q_vals, masks):
    """Argmax só sobre as ações seguras."""
    return q_vals.masked_fill(~masks, -float("inf")).argmax(dim=1)


# 2. Memória de replay em arrays numpy (amostragem O(1), ao contrário do deque)
class ReplayBuffer:
    def __init__(self, capacity, state_size):
        self.capacity = capacity
        self.s = np.zeros((capacity, state_size), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.ns = np.zeros((capacity, state_size), dtype=np.float32)
        self.d = np.zeros(capacity, dtype=bool)
        self.nm = np.zeros((capacity, 3), dtype=bool)
        self.pos = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done, next_mask):
        i = self.pos
        self.s[i], self.a[i], self.r[i] = state, action, reward
        self.ns[i], self.d[i], self.nm[i] = next_state, done, next_mask
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        to_t = lambda x: torch.from_numpy(x[idx]).to(DEVICE)
        return to_t(self.s), to_t(self.a), to_t(self.r), to_t(self.ns), to_t(self.d), to_t(self.nm)

    def __len__(self):
        return self.size


# 3. Agente Contínuo e Persistente (Double DQN + máscara de ações seguras)
class ContinuousAgent:
    def __init__(self):
        self.n_games = 0
        self.record = 0
        self.total_steps = 0
        self.train_steps = 0
        self.memory = ReplayBuffer(MAX_MEMORY, STATE_SIZE)

        self.model = Advanced_DQN().to(DEVICE)
        self.target_model = Advanced_DQN().to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.SmoothL1Loss()  # Huber loss é muito mais estável que MSE

        self.load_checkpoint()
        self.update_target_network()

    @property
    def epsilon(self):
        frac = min(self.total_steps / EPS_DECAY_STEPS, 1.0)
        return EPS_START + frac * (EPS_END - EPS_START)

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_checkpoint(self):
        os.makedirs("model", exist_ok=True)
        checkpoint = {
            'state_size': STATE_SIZE,
            'n_games': self.n_games,
            'record': self.record,
            'total_steps': self.total_steps,
            'train_steps': self.train_steps,
            'model_state': self.model.state_dict(),
            'target_state': self.target_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }
        torch.save(checkpoint, LATEST_PATH)

    def save_best(self, score):
        os.makedirs("model", exist_ok=True)
        torch.save({
            'state_size': STATE_SIZE,
            'record': score,
            'n_games': self.n_games,
            'model_state': self.model.state_dict(),
        }, BEST_PATH)

    def load_checkpoint(self):
        if not os.path.exists(LATEST_PATH):
            print("-> Nenhum checkpoint encontrado. A iniciar cérebro do zero.")
            return
        checkpoint = torch.load(LATEST_PATH, map_location=DEVICE)
        if checkpoint.get('state_size') != STATE_SIZE:
            print(f"-> {LATEST_PATH} tem um estado incompatível. A iniciar cérebro do zero.")
            return
        print(f"-> Checkpoint encontrado! A carregar cérebro existente de {LATEST_PATH}...")
        self.model.load_state_dict(checkpoint['model_state'])
        self.target_model.load_state_dict(checkpoint['target_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.n_games = checkpoint['n_games']
        self.record = checkpoint['record']
        self.total_steps = checkpoint['total_steps']
        self.train_steps = checkpoint['train_steps']
        print(f"-> Retomando a partir do Jogo {self.n_games} com Recorde de {self.record} (Epsilon: {self.epsilon:.3f})")

    def get_actions(self, states, masks):
        """Escolhe uma ação por ambiente. A exploração também só escolhe ações seguras."""
        with torch.no_grad():
            q_vals = self.model(torch.from_numpy(states).to(DEVICE))
            moves = masked_argmax(q_vals, torch.from_numpy(masks).to(DEVICE)).cpu().numpy()

        eps = self.epsilon
        for i in range(len(moves)):
            if random.random() < eps:
                moves[i] = random.choice(np.flatnonzero(masks[i]))
        return moves

    def train_step(self):
        if len(self.memory) < max(BATCH_SIZE, WARMUP):
            return

        s, a, r, ns, d, nm = self.memory.sample(BATCH_SIZE)

        # Q atual previsto
        q_eval = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Double DQN: a rede online escolhe a próxima ação (só entre as seguras),
        # a Target Network avalia-a
        with torch.no_grad():
            next_a = masked_argmax(self.model(ns), nm)
            q_next = self.target_model(ns).gather(1, next_a.unsqueeze(1)).squeeze(1)
            q_target = r + (~d) * GAMMA * q_next

        loss = self.criterion(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % TARGET_SYNC == 0:
            self.update_target_network()


def train():
    agent = ContinuousAgent()
    envs = [SnakeGameHeadless() for _ in range(NUM_ENVS)]
    obs = [get_state(g) for g in envs]
    recent_scores = deque(maxlen=100)
    last_print, last_steps = time.time(), agent.total_steps

    print(f"Treino contínuo iniciado em {DEVICE} com {NUM_ENVS} jogos em paralelo. "
          "Podes carregar Ctrl + C a qualquer momento sem perder progresso!")

    try:
        while True:
            states = np.stack([o[0] for o in obs])
            masks = np.stack([o[1] for o in obs])
            moves = agent.get_actions(states, masks)

            for i, game in enumerate(envs):
                old_dist = food_distance(game)
                action = [0, 0, 0]
                action[moves[i]] = 1
                reward, done, score = game.play_step(action)

                if done:
                    next_obs = terminal_state()
                else:
                    # Reward shaping baseado em potencial (phi = -dist): incentiva aproximar-se
                    # da comida sem mudar a política ótima, ao contrário de premiar sobrevivência
                    if reward == 0:
                        reward = SHAPING * (old_dist - GAMMA * food_distance(game))
                    next_obs = get_state(game)

                agent.memory.push(states[i], moves[i], reward, next_obs[0], done, next_obs[1])

                if done:
                    agent.n_games += 1
                    recent_scores.append(score)

                    if score > agent.record:
                        agent.record = score
                        agent.save_best(score)
                        print(f"⭐ NOVO RECORDE: {agent.record} pontos! Guardado em {BEST_PATH}.")

                    if agent.n_games % 100 == 0:
                        agent.save_checkpoint()

                    if agent.n_games % 20 == 0:
                        now = time.time()
                        sps = (agent.total_steps - last_steps) / max(now - last_print, 1e-6)
                        last_print, last_steps = now, agent.total_steps
                        avg = sum(recent_scores) / len(recent_scores)
                        print(f"Jogo {agent.n_games:5d} | Score: {score:4d} | Recorde: {agent.record:4d} | "
                              f"Média(100): {avg:7.2f} | Epsilon: {agent.epsilon:.3f} | "
                              f"Passos: {agent.total_steps:,} ({sps:.0f}/s)")

                    game.reset()
                    next_obs = get_state(game)

                obs[i] = next_obs

            agent.total_steps += NUM_ENVS
            for _ in range(NUM_ENVS // TRAIN_EVERY):
                agent.train_step()

    except KeyboardInterrupt:
        print("\n[Ctrl + C Detetado] A guardar checkpoint atual antes de sair...")
        agent.save_checkpoint()
        print("Guardado com sucesso! Podes fechar o terminal e voltar a correr quando quiseres.")


if __name__ == "__main__":
    train()
