import pickle
import numpy as np
import matplotlib.pyplot as plt

# === Load Data ===
try:
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
    rewards = np.load("rewards.npy")
except FileNotFoundError:
    print("⚠️ Missing data files. Run main.py first to generate q_table.pkl and rewards.npy.")
    exit()
except EOFError:
    print("⚠️ q_table.pkl seems empty or corrupted. Try re-running main.py.")
    exit()

# === 1️⃣ Plot Reward Trend ===
plt.figure(figsize=(8,5))
plt.plot(rewards, label="Reward per Step", linewidth=2)
plt.xlabel("Simulation Step")
plt.ylabel("Reward")
plt.title("Reward Trend During Training")
plt.legend()
plt.grid(True)
plt.show()

# === 2️⃣ Visualize Q-table as Heatmap ===
if len(q_table) > 0:
    states = list(q_table.keys())
    actions = list(next(iter(q_table.values())).keys())
    q_matrix = np.array([[q_table[s][a] for a in actions] for s in states])

    plt.figure(figsize=(8,6))
    plt.imshow(q_matrix, cmap="viridis", aspect="auto")
    plt.colorbar(label="Q-value")
    plt.xticks(range(len(actions)), actions)
    plt.yticks(range(len(states)), [f"S{i}" for i in range(len(states))])
    plt.title("Q-table Heatmap")
    plt.xlabel("Actions")
    plt.ylabel("States")
    plt.show()
else:
    print("⚠️ Q-table is empty. The agent might not have interacted with multiple states yet.")
