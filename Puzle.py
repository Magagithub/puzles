# ============================================
# 0. IMPORTACIONES
# ============================================
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import threading
import time

# -------------------
# 8-PUZZLE LOGIC
# -------------------

goal_state = (0, 1, 2, 3, 4, 5, 6, 7, 8)
actions = ["up", "right", "down", "left"]

moves = {
    "up": -3,
    "right": 1,
    "down": 3,
    "left": -1
}

# Precomputamos posiciones objetivo de cada ficha para la heurística
goal_pos = {tile: idx for idx, tile in enumerate(goal_state)}


def valid_actions(state):
    hole = state.index(8)
    row, col = divmod(hole, 3)
    valid = []
    if row > 0:
        valid.append("up")
    if row < 2:
        valid.append("down")
    if col < 2:
        valid.append("right")
    if col > 0:
        valid.append("left")
    return valid


def manhattan_distance(state):
    """
    Heurística: suma de distancias Manhattan de cada ficha a su posición objetivo.
    No contamos el hueco (8).
    """
    dist = 0
    for idx, tile in enumerate(state):
        if tile == 8:
            continue
        r, c = divmod(idx, 3)
        gr, gc = divmod(goal_pos[tile], 3)
        dist += abs(r - gr) + abs(c - gc)
    return dist


def step_env(state, action):
    """
    Transición de entorno + reward shaping.
    - Si la acción es inválida: penalización.
    - Si es válida: se mueve el hueco, coste por paso,
      y se añade un término de recompensa según si nos acercamos o alejamos del objetivo
      usando la distancia Manhattan.
    """
    if action not in valid_actions(state):
        # Penalización fuerte por acción inválida
        return state, -5

    old_dist = manhattan_distance(state)

    hole = state.index(8)
    new_pos = hole + moves[action]
    s = list(state)
    s[hole], s[new_pos] = s[new_pos], s[hole]
    new_state = tuple(s)

    new_dist = manhattan_distance(new_state)

    # Coste base por paso
    reward = -1

    # Reward shaping: positivo si mejoramos la distancia, negativo si empeoramos
    reward += (old_dist - new_dist)

    # Bonus fuerte al llegar al objetivo
    if new_state == goal_state:
        reward += 50

    return new_state, reward


def shuffle_state(state, moves_count=20):
    """
    Baraja el puzzle aplicando 'moves_count' movimientos aleatorios válidos.
    """
    s = state
    for _ in range(moves_count):
        a = random.choice(valid_actions(s))
        s, _ = step_env(s, a)
    return s


# Tabla Q: para cada estado, diccionario acción->valor
Q = defaultdict(lambda: {a: 0.0 for a in actions})


def epsilon_greedy(state, epsilon):
    va = valid_actions(state)
    if random.random() < epsilon:
        return random.choice(va)
    qv = Q[state]
    return max(va, key=lambda a: qv[a])


def q_update(s, a, r, s2, alpha, gamma):
    best_next = max(Q[s2].values())
    Q[s][a] = Q[s][a] + alpha * (r + gamma * best_next - Q[s][a])


def train_q_learning(
    episodes=20000,
    alpha=0.2,
    gamma=0.99,
    eps_start=1.0,
    eps_min=0.05,
    max_steps=200,
    shuffle_depth=20
):
    """
    Entrena el agente con Q-Learning.
    - Más episodios y épsilon decreciente para una exploración-explotación razonable.
    - Barajado no demasiado profundo para que aprenda primero estados alcanzables.
    """
    global Q
    Q = defaultdict(lambda: {a: 0.0 for a in actions})

    steps_log = []
    success_log = []

    epsilon = eps_start

    for ep in range(episodes):
        state = shuffle_state(goal_state, shuffle_depth)
        steps = 0
        success = 0

        for _ in range(max_steps):
            a = epsilon_greedy(state, epsilon)
            s2, r = step_env(state, a)
            q_update(state, a, r, s2, alpha, gamma)

            state = s2
            steps += 1

            if state == goal_state:
                success = 1
                break

        steps_log.append(steps)
        success_log.append(success)

        # Decaimiento del epsilon (no baja de eps_min)
        epsilon = max(eps_min, epsilon * 0.9995)

    return steps_log, success_log


def solve_with_Q_from_state(initial_state, max_steps=200, epsilon_exec=0.05):
    """
    Usa la tabla Q aprendida para resolver desde un estado concreto.
    - Política casi-greedy (epsilon pequeño) para evitar bucles.
    - Sin visited: dejamos que max_steps ponga el límite.
    """
    state = initial_state
    path = [state]

    for _ in range(max_steps):
        if state == goal_state:
            break

        va = valid_actions(state)

        # Un poquito de exploración en ejecución
        if random.random() < epsilon_exec:
            action = random.choice(va)
        else:
            qvals = Q[state]
            action = max(va, key=lambda a: qvals[a])

        new_state, _ = step_env(state, action)
        state = new_state
        path.append(state)

        if state == goal_state:
            break

    return path


# =====================================
# GUI (Tkinter)
# =====================================

class PuzzleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("8-Puzzle con Q-Learning - Miguel Magariño")

        self.frame = tk.Frame(root)
        self.frame.pack()

        # botones
        tk.Button(self.frame, text="Cargar imagen", command=self.load_image).grid(row=0, column=0)
        tk.Button(self.frame, text="Barajar", command=self.shuffle_puzzle).grid(row=0, column=1)
        tk.Button(self.frame, text="Entrenar Q-Learning", command=self.train_agent).grid(row=0, column=2)
        tk.Button(self.frame, text="Resolver", command=self.solve_puzzle).grid(row=0, column=3)

        # canvas puzzle
        self.canvas_cells = []
        self.image_tiles = []
        self.current_state = goal_state

        self.grid_frame = tk.Frame(self.frame)
        self.grid_frame.grid(row=1, column=0, columnspan=4)

    # -------------------------------
    # Cargar imagen / dividirla
    # -------------------------------
    def load_image(self):
        path = filedialog.askopenfilename()
        if not path:
            return

        img = Image.open(path).convert("RGB")
        w, h = img.size
        tile_w = w // 3
        tile_h = h // 3

        self.image_tiles = []

        for r in range(3):
            for c in range(3):
                crop = img.crop((c * tile_w, r * tile_h, c * tile_w + tile_w, r * tile_h + tile_h))
                crop = crop.resize((120, 120))
                self.image_tiles.append(ImageTk.PhotoImage(crop))

        # preparar canvas
        for widget in self.grid_frame.winfo_children():
            widget.destroy()

        self.canvas_cells = []
        for i in range(9):
            cv = tk.Canvas(self.grid_frame, width=120, height=120, bg="white")
            cv.grid(row=i // 3, column=i % 3)
            self.canvas_cells.append(cv)

        self.current_state = goal_state
        self.draw_puzzle(self.current_state)

    def draw_puzzle(self, state):
        for i, tile in enumerate(state):
            canvas = self.canvas_cells[i]
            canvas.delete("all")
            if tile == 8 or not self.image_tiles:
                canvas.create_rectangle(0, 0, 120, 120, fill="white")
            else:
                canvas.create_image(0, 0, anchor="nw", image=self.image_tiles[tile])

    # -------------------------------
    # Barajar puzzle
    # -------------------------------
    def shuffle_puzzle(self):
        # Usamos un barajado moderado para aumentar la probabilidad
        # de que el estado esté en el "espacio aprendido".
        self.current_state = shuffle_state(goal_state, moves_count=20)
        self.draw_puzzle(self.current_state)

    # -------------------------------
    # Entrenar Q-Learning
    # -------------------------------
    def train_agent(self):
        def run_training():
            steps, success = train_q_learning()

            # Gráfica de pasos por episodio
            plt.figure(figsize=(6, 3))
            plt.plot(steps)
            plt.title("Pasos por episodio")
            plt.grid()
            plt.tight_layout()
            plt.show()

            # Gráfica de éxito por episodio
            plt.figure(figsize=(6, 3))
            plt.plot(success)
            plt.title("Éxito por episodio (1=exito, 0=fracaso)")
            plt.grid()
            plt.tight_layout()
            plt.show()

        threading.Thread(target=run_training, daemon=True).start()

    # -------------------------------
    # Resolver puzzle con Q
    # -------------------------------
    def solve_puzzle(self):
        def animate():
            path = solve_with_Q_from_state(self.current_state, max_steps=200, epsilon_exec=0.05)
            for state in path:
                self.draw_puzzle(state)
                time.sleep(0.3)
            # Actualizamos estado actual por si no quedó exacto
            self.current_state = path[-1]

        threading.Thread(target=animate, daemon=True).start()


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = PuzzleApp(root)
    root.mainloop()
