# ============================================
# 0. IMPORTACIONES
# ============================================
from google.colab import files #nuevo comentario conflicto
import ipywidgets as widgets
from IPython.display import display, clear_output
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict

plt.rcParams["figure.figsize"] = (4,4)

# ============================================
# 1. LÓGICA DEL 8-PUZZLE + Q-LEARNING
# ============================================
# Representación:
# - Estado = tupla de 9 enteros (0..8)
# - 0..7 -> fichas numeradas
# - 8   -> hueco (casilla vacía)
# - goal_state -> estado objetivo
#
# Visualmente mostraremos fichas 1..8 y el hueco en blanco.

goal_state = (0,1,2,3,4,5,6,7,8)   # estado objetivo (hueco abajo a la derecha)
actions = ["up", "right", "down", "left"]

# desplazamientos del índice del hueco en la lista lineal
moves = {
    "up":    -3,
    "right":  1,
    "down":   3,
    "left":  -1
}

def valid_actions(state):
    """
    Devuelve la lista de acciones válidas (que no sacan el hueco fuera del tablero).
    """
    hole_pos = state.index(8)  # 8 = hueco
    row, col = divmod(hole_pos, 3)
    valid = []
    if row > 0: valid.append("up")
    if row < 2: valid.append("down")
    if col < 2: valid.append("right")
    if col > 0: valid.append("left")
    return valid

def step_env(state, action):
    """
    Aplica una acción sobre el estado del puzzle.
    Devuelve (new_state, reward).
    Recompensas:
      - movimiento válido normal: -1
      - si alcanzamos el estado objetivo: +100
      - si la acción no es válida: castigo fuerte (-5)
    """
    if action not in valid_actions(state):
        return state, -5

    hole_pos = state.index(8)
    new_pos = hole_pos + moves[action]

    s_list = list(state)
    # intercambiamos el hueco con la ficha destino
    s_list[hole_pos], s_list[new_pos] = s_list[new_pos], s_list[hole_pos]
    new_state = tuple(s_list)

    if new_state == goal_state:
        return new_state, 100
    else:
        return new_state, -1

def shuffle_state(state, moves_count=30):
    """
    Baraja el estado aplicando una cadena de movimientos válidos aleatorios.
    Garantiza que el estado resultante sea alcanzable (resoluble).
    """
    s = state
    for _ in range(moves_count):
        va = valid_actions(s)
        a = random.choice(va)
        s, _ = step_env(s, a)
    return s

# Q-Table: Q[s][a] -> valor Q del estado s con acción a
Q = defaultdict(lambda: {a: 0.0 for a in actions})


def epsilon_greedy(state, epsilon):
    """
    Política ε-greedy:
      - con prob. epsilon -> acción aleatoria (exploración)
      - si no -> mejor acción según Q (explotación)
    Siempre dentro de las acciones válidas.
    """
    va = valid_actions(state)
    # por seguridad, si no hubiera acciones válidas
    if not va:
        return random.choice(actions)

    if random.random() < epsilon:
        return random.choice(va)

    qvals = Q[state]
    best_a = max(va, key=lambda a: qvals[a])
    return best_a


def q_update(s, a, r, s2, alpha, gamma):
    """
    Ecuación de actualización Q-Learning:
    Q(s,a) <- Q(s,a) + α [ r + γ max_a' Q(s', a') - Q(s,a) ]
    """
    best_next = max(Q[s2].values())
    Q[s][a] = Q[s][a] + alpha * (r + gamma * best_next - Q[s][a])


def train_q_learning(episodes=5000, alpha=0.2, gamma=0.95,
                     epsilon_start=0.3, max_steps=200):
    """
    Entrena el agente Q-Learning sobre el 8-puzzle abstracto.
    Devuelve:
      - steps_log: nº de pasos por episodio
      - success_log: 1 si resuelve, 0 si no
    """
    global Q
    Q = defaultdict(lambda: {a: 0.0 for a in actions})  # reiniciamos la Q-Table

    steps_log = []
    success_log = []

    epsilon = epsilon_start

    for ep in range(episodes):
        # Partimos de un estado barajado alcanzable
        state = shuffle_state(goal_state, moves_count=30)
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
            # No llegó a la solución dentro del límite de pasos
            success_log.append(0)

        steps_log.append(steps)

        # Reducimos epsilon poco a poco para ir explotando más lo aprendido
        epsilon = max(0.01, epsilon * 0.999)

    return steps_log, success_log


def solve_with_Q_from_state(initial_state, max_steps=200):
    """
    Intenta resolver el puzzle desde 'initial_state' usando la Q-Table.
    Si detecta un ciclo, intenta una acción alternativa válida.
    """
    state = initial_state
    path = [state]
    visited = set()

    for _ in range(max_steps):

        if state == goal_state:
            break

        if state in visited:
            # Detectado ciclo → intentar escape
            print("⚠ Ciclo detectado → intentando acción alternativa...")

            va = valid_actions(state)
            # Quitamos la mejor opción para forzar opción alternativa
            qvals = Q[state]

            # Acciones ordenadas de mejor a peor
            ordered = sorted(va, key=lambda a: qvals[a], reverse=True)

            escape_done = False
            for alt in ordered[1:]:  # probamos alternativas, no la mejor
                new_state, _ = step_env(state, alt)
                if new_state not in visited:
                    state = new_state
                    path.append(state)
                    escape_done = True
                    break

            if not escape_done:
                print("❌ No hay escape posible, el agente está atrapado.")
                break

        else:
            visited.add(state)

            # Acción greedy normal
            va = valid_actions(state)
            qvals = Q[state]
            best_a = max(va, key=lambda a: qvals[a])

            new_state, _ = step_env(state, best_a)
            state = new_state
            path.append(state)

    return path


# ============================================
# 2. PARTE VISUAL (IMAGEN + IPYWIDGETS)
# ============================================

original_image = None   # imagen completa
img_slices = []         # 9 trozos (3x3)
current_state = goal_state  # estado actual del puzzle (lo que se ve)

upload_button = widgets.Button(description="📷 Subir imagen")
shuffle_button = widgets.Button(description="🔀 Barajar", disabled=True)
train_button   = widgets.Button(description="🤖 Entrenar Q-Learning", disabled=True)
solve_button   = widgets.Button(description="✅ Resolver con Q", disabled=True)

output = widgets.Output()


def show_puzzle(state):
    """
    Dibuja el estado del puzzle con la imagen dividida.
    - Si tile == 8 -> hueco (se dibuja un cuadrado blanco)
    - Si no -> se dibuja el trozo de imagen correspondiente.
    """
    plt.figure(figsize=(4,4))
    for i, tile in enumerate(state):
        plt.subplot(3, 3, i+1)
        if tile == 8 or len(img_slices) != 9:
            # Hueco o aún no hay imagen: recuadro blanco
            plt.imshow(np.ones((50,50,3)))
            plt.title(" ")
        else:
            plt.imshow(img_slices[tile])
            # Mostramos numeración humana 1..8
            plt.title(str(tile + 1))
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def load_image(path):
    """
    Carga la imagen de 'path', la convierte a RGB y la divide en 9 trozos.
    """
    global original_image, img_slices, current_state
    original_image = Image.open(path).convert("RGB")

    w, h = original_image.size
    tile_w = w // 3
    tile_h = h // 3

    img_slices = []
    for r in range(3):
        for c in range(3):
            left = c * tile_w
            top = r * tile_h
            crop = original_image.crop((left, top, left + tile_w, top + tile_h))
            img_slices.append(crop)

    # Empezamos en el estado objetivo (puzzle resuelto)
    current_state = goal_state

    with output:
        clear_output()
        print("✅ Imagen cargada y puzzle preparado (estado objetivo).")
        show_puzzle(current_state)

    # Activamos botones
    shuffle_button.disabled = False
    train_button.disabled = False
    solve_button.disabled = False


def on_upload_clicked(b):
    with output:
        clear_output()
        print("Selecciona una imagen desde tu ordenador...")
    uploaded = files.upload()
    if len(uploaded) == 0:
        with output:
            print("No se ha seleccionado ninguna imagen.")
        return
    file_name = list(uploaded.keys())[0]
    load_image(file_name)


def on_shuffle_clicked(b):
    """
    Baraja el puzzle aplicando movimientos válidos,
    asegurando que el estado sea resoluble.
    """
    global current_state
    current_state = shuffle_state(goal_state, moves_count=40)
    with output:
        clear_output()
        print("🔀 Puzzle barajado (estado alcanzable):")
        show_puzzle(current_state)


def on_train_clicked(b):
    """
    Entrena el agente Q-Learning sobre muchos estados aleatorios del 8-puzzle.
    Muestra gráficas de:
      - pasos por episodio
      - tasa de éxito
    """
    with output:
        clear_output()
        print("🤖 Entrenando agente Q-Learning... puede tardar unos segundos.")

    steps_log, success_log = train_q_learning(
        episodes=5000, alpha=0.2, gamma=0.95,
        epsilon_start=0.3, max_steps=200
    )

    with output:
        print("✅ Entrenamiento completado.")
        # Gráfica de pasos
        plt.figure(figsize=(5,3))
        plt.plot(steps_log)
        plt.xlabel("Episodio")
        plt.ylabel("Pasos")
        plt.title("Pasos por episodio durante el entrenamiento")
        plt.grid(True)
        plt.show()

        # Gráfica de éxito
        plt.figure(figsize=(5,3))
        plt.plot(success_log)
        plt.xlabel("Episodio")
        plt.ylabel("Éxito (1) / Fracaso (0)")
        plt.title("Tasa de éxito por episodio")
        plt.grid(True)
        plt.show()

        print(f"Éxito en {sum(success_log)} de {len(success_log)} episodios.")


def on_solve_clicked(b):
    """
    Intenta resolver el estado actual del puzzle usando la Q-Table entrenada.
    Dibuja cada estado del camino.
    """
    global current_state
    with output:
        clear_output()
        print("✅ Resolviendo puzzle con la Q-Table (política codiciosa)...")

    path = solve_with_Q_from_state(current_state, max_steps=200)

    with output:
        if path[-1] != goal_state:
            print("⚠ El agente NO ha conseguido llegar al estado objetivo con la Q actual.")
            print(f"Longitud del camino generado: {len(path)} estados.")
        else:
            print(f"🎉 Puzzle resuelto en {len(path)-1} movimientos.")
        # Mostramos todos los estados del camino
        for i, st in enumerate(path):
            print(f"Paso {i}:")
            show_puzzle(st)
        current_state = path[-1]


# Asociamos callbacks
upload_button.on_click(on_upload_clicked)
shuffle_button.on_click(on_shuffle_clicked)
train_button.on_click(on_train_clicked)
solve_button.on_click(on_solve_clicked)

# Mostramos interfaz
buttons_box = widgets.HBox([upload_button, shuffle_button, train_button, solve_button])
display(buttons_box, output)
