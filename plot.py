import torch
import pandas as pd
import matplotlib.pyplot as plt

regret_ego = torch.zeros((20, 30))
regret_local = torch.zeros((20, 30))
regret_random = torch.zeros((20, 30))
for i in range(20):
    regret_ego[i,:] = torch.load(f"../logs/synthetic/ei_ego_network_1/{str(i).zfill(4)}_ei_ego_network_1.pt")["regret"].flatten()[:30]
    regret_local[i,:] = torch.load(f"../logs/synthetic/local_search/{str(i).zfill(4)}_local_search.pt")["regret"].flatten()[:30]
    regret_random[i,:] = torch.load(f"../logs/synthetic/random/{str(i).zfill(4)}_random.pt")["regret"].flatten()[:30]

x = range(30)

mean_random = pd.DataFrame(regret_random.numpy()).cummin(axis = 1).mean(axis=0)
std_random = pd.DataFrame(regret_random.numpy()).cummin(axis = 1).std(axis=0)
plt.plot(x, pd.DataFrame(regret_random.numpy()).cummin(axis = 1).mean(axis=0), label="random")
plt.fill_between(x, mean_random - std_random, mean_random + std_random, color='blue', alpha=0.2)

mean_ego = pd.DataFrame(regret_ego.numpy()).cummin(axis = 1).mean(axis=0)
std_ego = pd.DataFrame(regret_ego.numpy()).cummin(axis = 1).std(axis=0)
plt.plot(x, pd.DataFrame(regret_ego.numpy()).cummin(axis = 1).mean(axis=0), label="ego")
plt.fill_between(x, mean_ego - std_ego, mean_ego + std_ego, color='red', alpha=0.2)

mean_local = pd.DataFrame(regret_local.numpy()).cummin(axis = 1).mean(axis=0)
std_local = pd.DataFrame(regret_local.numpy()).cummin(axis = 1).std(axis=0)
plt.plot(x, pd.DataFrame(regret_local.numpy()).cummin(axis = 1).mean(axis=0), label="local")
plt.fill_between(x, mean_local - std_local, mean_local + std_local, color='green', alpha=0.2)


plt.legend()