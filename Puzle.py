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

goal_state = (0,1,2,3,4,5,6,7,8)
actions = ["up", "right", "down", "left"]

moves = {
    "up": -3,
    "right": 1,
    "down": 3,
    "left": -1
}

def valid_actions(state):
    hole = state.index(8)
    row, col = divmod(hole, 3)
    valid = []
    if row > 0: valid.append("up")
    if row < 2: valid.append("down")
    if col < 2: valid.append("right")
    if col > 0: valid.append("left")
    return valid

def step_env(state, action):
    if action not in valid_actions(state):
        return state, -5
    hole = state.index(8)
    new_pos = hole + moves[action]
    s = list(state)
    s[hole], s[new_pos] = s[new_pos], s[hole]
    new_state = tuple(s)
    reward = 100 if new_state == goal_state else -1
    return new_state, reward

def shuffle_state(state, moves_count=40):
    s = state
    for _ in range(moves_count):
        a = random.choice(valid_actions(s))
        s, _ = step_env(s, a)
    return s

Q = defaultdict(lambda: {a: 0.0 for a in actions})

def epsilon_greedy(state, epsilon):
    va = valid_actions(state)
    if random.random() < epsilon:
        return random.choice(va)
    qv = Q[state]
    return max(va, key=lambda a: qv[a])

def q_update(s, a, r, s2, alpha, gamma):
    best_next = max(Q[s2].values())
    Q[s][a] = Q[s][a] + alpha*(r + gamma*best_next - Q[s][a])

def train_q_learning(episodes=5000, alpha=0.2, gamma=0.95, eps=0.3, max_steps=200):
    global Q
    Q = defaultdict(lambda: {a: 0.0 for a in actions})
    steps_log = []
    success_log = []
    epsilon = eps

    for _ in range(episodes):
        state = shuffle_state(goal_state, 40)
        steps = 0

        for _ in range(max_steps):
            a = epsilon_greedy(state, epsilon)
            s2, r = step_env(state, a)
            q_update(state, a, r, s2, alpha, gamma)
            state = s2
            steps += 1
            if state == goal_state:
                success_log.append(1)
                break
        else:
            success_log.append(0)

        steps_log.append(steps)
        epsilon = max(0.01, epsilon * 0.999)

    return steps_log, success_log

def solve_with_Q_from_state(initial_state, max_steps=200):
    state = initial_state
    path = [state]
    visited = set()

    for _ in range(max_steps):

        if state == goal_state:
            break

        if state in visited:
            # escape from cycle
            va = valid_actions(state)
            qvals = Q[state]
            ordered = sorted(va, key=lambda a: qvals[a], reverse=True)

            escape = False
            for alt in ordered[1:]:
                new_state, _ = step_env(state, alt)
                if new_state not in visited:
                    state = new_state
                    path.append(state)
                    escape = True
                    break

            if not escape:
                break
        else:
            visited.add(state)
            va = valid_actions(state)
            qvals = Q[state]
            best = max(va, key=lambda a: qvals[a])
            new_state, _ = step_env(state, best)
            state = new_state
            path.append(state)

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
                crop = img.crop((c*tile_w, r*tile_h, c*tile_w+tile_w, r*tile_h+tile_h))
                crop = crop.resize((120, 120))
                self.image_tiles.append(ImageTk.PhotoImage(crop))

        # preparar canvas
        for widget in self.grid_frame.winfo_children():
            widget.destroy()

        self.canvas_cells = []
        for i in range(9):
            cv = tk.Canvas(self.grid_frame, width=120, height=120, bg="white")
            cv.grid(row=i//3, column=i%3)
            self.canvas_cells.append(cv)

        self.current_state = goal_state
        self.draw_puzzle(self.current_state)

    def draw_puzzle(self, state):
        for i, tile in enumerate(state):
            canvas = self.canvas_cells[i]
            canvas.delete("all")
            if tile == 8 or not self.image_tiles:
                canvas.create_rectangle(0,0,120,120, fill="white")
            else:
                canvas.create_image(0,0, anchor="nw", image=self.image_tiles[tile])

    # -------------------------------
    # Barajar puzzle
    # -------------------------------
    def shuffle_puzzle(self):
        self.current_state = shuffle_state(goal_state, 40)
        self.draw_puzzle(self.current_state)

    # -------------------------------
    # Entrenar Q-Learning
    # -------------------------------
    def train_agent(self):
        def run_training():
            steps, success = train_q_learning()
            plt.figure(figsize=(6,3))
            plt.plot(steps)
            plt.title("Pasos por episodio")
            plt.grid()
            plt.show()

            plt.figure(figsize=(6,3))
            plt.plot(success)
            plt.title("Éxito por episodio")
            plt.grid()
            plt.show()

        threading.Thread(target=run_training).start()

    # -------------------------------
    # Resolver puzzle con Q
    # -------------------------------
    def solve_puzzle(self):
        def animate():
            path = solve_with_Q_from_state(self.current_state)
            for state in path:
                self.draw_puzzle(state)
                time.sleep(0.3)
            self.current_state = path[-1]

        threading.Thread(target=animate).start()


# ==================================================
# MAIN
# ==================================================

root = tk.Tk()
app = PuzzleApp(root)
root.mainloop()
