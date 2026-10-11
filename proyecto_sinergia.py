%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from IPython.display import HTML

class StatisticalAgent:
    def __init__(self, agent_id, start_x, start_y):
        self.id = agent_id
        self.x = start_x
        self.y = start_y
        self.dopamine = 0.5   
        self.serotonin = 0.5  
        self.cortisol = 0.1   
        self.energy = 1.0     
        self.state = "CALMA"
        
        self.memory_path = []
        self.max_memory_len = 15

    def decide_and_move(self, rewards_list, danger_pos):
        self.memory_path.append((self.x, self.y))
        if len(self.memory_path) > self.max_memory_len:
            self.memory_path.pop(0)

        self.energy = max(0.0, self.energy - 0.015)
        
        dx = np.random.uniform(-0.5, 0.5)
        dy = np.random.uniform(-0.5, 0.5)
        
        if self.cortisol > 0.5 or np.hypot(self.x - danger_pos[0], self.y - danger_pos[1]) < 3.0:
            dx += (self.x - danger_pos[0]) * 0.45
            dy += (self.y - danger_pos[1]) * 0.45
            self.state = "HUIDA"
        elif self.energy < 0.5 or self.dopamine < 0.3:
            target = rewards_list[0]
            min_dist = np.hypot(self.x - target[0], self.y - target[1])
            for r in rewards_list:
                dist = np.hypot(self.x - r[0], self.y - r[1])
                if dist < min_dist:
                    min_dist = dist
                    target = r
            dx += (target[0] - self.x) * 0.25
            dy += (target[1] - self.y) * 0.25
            self.state = "BUSCANDO"
        else:
            self.state = "PATRULLA"

        self.x = np.clip(self.x + dx, 0, 10)
        self.y = np.clip(self.y + dy, 0, 10)

        near_reward = any(np.hypot(self.x - r[0], self.y - r[1]) < 1.0 for r in rewards_list)
        dist_danger = np.hypot(self.x - danger_pos[0], self.y - danger_pos[1])

        if near_reward:  
            self.dopamine = min(1.0, self.dopamine + 0.35)
            self.serotonin = min(1.0, self.serotonin + 0.2)
            self.cortisol = max(0.0, self.cortisol - 0.25)
            self.energy = min(1.0, self.energy + 0.5)  
        elif dist_danger < 1.5:  
            self.cortisol = min(1.0, self.cortisol + 0.4)
            self.dopamine = max(0.0, self.dopamine - 0.2)
        else:  
            self.dopamine = max(0.1, self.dopamine - 0.02)
            self.cortisol = max(0.0, self.cortisol - 0.02)

        if self.energy < 0.2:
            self.cortisol = min(1.0, self.cortisol + 0.2)
            self.state = "FATIGA"

num_agents = 3
agents = [StatisticalAgent(i, np.random.uniform(1, 4), np.random.uniform(1, 4)) for i in range(num_agents)]

rewards_list = [[8.0, 8.0], [2.0, 8.0]]
danger_pos = [5.0, 5.0]
danger_velocity = [0.1, 0.05]

history_dopamine = []
history_cortisol = []
history_energy = []

fig = plt.figure(figsize=(15, 4.5))
ax_map = plt.subplot(1, 3, 1)
ax_chem = plt.subplot(1, 3, 2)
ax_trend = plt.subplot(1, 3, 3)

colors_agents = ['#1f77b4', '#ff7f0e', '#2ca02c']

def animate(frame):
    global danger_pos
    danger_pos[0] += danger_velocity[0]
    danger_pos[1] += danger_velocity[1]
    if danger_pos[0] <= 1.0 or danger_pos[0] >= 9.0: danger_velocity[0] *= -1
    if danger_pos[1] <= 1.0 or danger_pos[1] >= 9.0: danger_velocity[1] *= -1

    ax_map.clear()
    ax_map.set_xlim(0, 10)
    ax_map.set_ylim(0, 10)
    ax_map.set_title(f"Entorno | Frame {frame}", fontsize=9, fontweight='bold')
    
    for r in rewards_list:
        ax_map.scatter(r[0], r[1], c='green', s=140, marker='s')
    ax_map.scatter(danger_pos[0], danger_pos[1], c='red', s=160, marker='X')

    avg_dop, avg_ser, avg_cor, avg_ene = 0, 0, 0, 0

    for idx, agent in enumerate(agents):
        agent.decide_and_move(rewards_list, danger_pos)
        avg_dop += agent.dopamine
        avg_ser += agent.serotonin
        avg_cor += agent.cortisol
        avg_ene += agent.energy

        if agent.memory_path:
            mx, my = zip(*agent.memory_path)
            ax_map.plot(mx, my, color=colors_agents[idx], linestyle='--', alpha=0.4)

        col = 'red' if agent.cortisol > 0.5 else colors_agents[idx]
        ax_map.scatter(agent.x, agent.y, c=col, s=100, edgecolors='black')
    ax_map.grid(True, linestyle='--', alpha=0.5)

    avg_dop /= num_agents
    avg_ser /= num_agents
    avg_cor /= num_agents
    avg_ene /= num_agents

    history_dopamine.append(avg_dop)
    history_cortisol.append(avg_cor)
    history_energy.append(avg_ene)

    ax_chem.clear()
    ax_chem.set_ylim(0, 1.0)
    ax_chem.set_title("Niveles Actuales", fontsize=9, fontweight='bold')
    metrics = ['Dop.', 'Ser.', 'Cor.', 'Ene.']
    values = [avg_dop, avg_ser, avg_cor, avg_ene]
    bar_colors = ['#3498db', '#2ecc71', '#e74c3c', '#f1c40f']
    ax_chem.bar(metrics, values, color=bar_colors, width=0.5)

    ax_trend.clear()
    ax_trend.set_ylim(0, 1.0)
    ax_trend.set_title("Evolución Histórica", fontsize=9, fontweight='bold')
    ax_trend.plot(history_dopamine, label='Dopamina', color='#3498db', linewidth=2)
    ax_trend.plot(history_cortisol, label='Cortisol', color='#e74c3c', linewidth=2)
    ax_trend.plot(history_energy, label='Energía', color='#f1c40f', linewidth=2)
    ax_trend.legend(loc='upper right', fontsize=6)
    ax_trend.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

ani = animation.FuncAnimation(fig, animate, frames=90, interval=250, repeat=True)
plt.close(fig) 

HTML(ani.to_jshtml())
