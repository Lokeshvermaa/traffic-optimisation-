import os
import sys
import traci
import numpy as np
import pickle
from q_learning import QLearningAgent
from utils import get_state, get_reward

# === SUMO PATH SETUP ===
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

sumo_cfg = "sumo_simulation/hello.sumocfg"
sumo_cmd = ["sumo-gui", "-c", sumo_cfg]

# === Start TraCI ===
traci.start(sumo_cmd)
print("SUMO simulation started with TraCI connection.")

# === Q-learning setup ===
agent = QLearningAgent(actions=["green", "yellow", "red"])
prev_state, prev_action = None, None
rewards = []

# === Simulation loop ===
for step in range(500):        # keep it small for quick test
    traci.simulationStep()
    curr_state = get_state(traci)
    action = agent.choose_action(curr_state)

    # Apply chosen light phase
    for tl_id in traci.trafficlight.getIDList():
        if action == "green":
            state_str = "G" * 8 + "r" * 8
        elif action == "yellow":
            state_str = "y" * 8 + "r" * 8
        else:
            state_str = "r" * 8 + "G" * 8
        traci.trafficlight.setRedYellowGreenState(tl_id, state_str)

    traci.simulationStep()
    reward = get_reward(traci)
    rewards.append(reward)

    # Learn only if previous step exists
    if prev_state is not None and prev_action is not None:
        agent.learn(prev_state, prev_action, reward, curr_state)

    prev_state, prev_action = curr_state, action

    if step % 50 == 0:
        print(f"Step {step}: Action={action}, Reward={reward}")

traci.close()

# === Save data ===
with open("q_table.pkl", "wb") as f:
    pickle.dump(dict(agent.q_table), f)
np.save("rewards.npy", np.array(rewards))

print(f"✅ Simulation finished. Saved {len(agent.q_table)} Q-states and {len(rewards)} rewards.")
