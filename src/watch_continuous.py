import os
import torch
import pygame

from game import SnakeGameAI
from features import STATE_SIZE, get_state
from train_continuous import Advanced_DQN, masked_argmax, BEST_PATH, LATEST_PATH


def watch():
    path = BEST_PATH if os.path.exists(BEST_PATH) else LATEST_PATH
    if not os.path.exists(path):
        print(f"Erro: Checkpoint não encontrado em {BEST_PATH} nem em {LATEST_PATH}")
        return

    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get('state_size') != STATE_SIZE:
        print(f"Erro: {path} foi treinado com um estado diferente. Volta a treinar com train_continuous.py")
        return

    model = Advanced_DQN()
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    pygame.init()
    game = SnakeGameAI()

    print(f"A assistir ao modelo {path} (Recorde guardado: {checkpoint.get('record', 0)} | "
          f"Jogo de origem: {checkpoint.get('n_games', 0)})")

    while True:
        state, mask = get_state(game)

        with torch.no_grad():
            q_values = model(torch.from_numpy(state).unsqueeze(0))
            move_idx = masked_argmax(q_values, torch.from_numpy(mask).unsqueeze(0)).item()

        final_move = [0, 0, 0]
        final_move[move_idx] = 1

        reward, done, score = game.play_step(final_move)

        if done:
            print(f"Fim de jogo! Pontuação atingida: {score}")
            game.reset()


if __name__ == "__main__":
    watch()
