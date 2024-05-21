
import torch
from model import Model
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np

counter = 0
loss_container = []
observations_container = []
with open('loss.txt', 'r') as file:
    for line in file:
        if "Total" in line:
            split_string = line.split()
            loss_container.append(float(split_string[4]))
            observations_container.append(int(split_string[7]))
            counter += 1
            if counter == 176:
                observations = int(split_string[7])

print(len(loss_container))
print(counter)

def selective_load(source_dict, target):
    target.load_state_dict(source_dict)
    
def save_new_loss ():
    LOAD_PATH = "model/ModelData.tm"
    model = Model()
    LEARNING_RATE = 0.001
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    checkpoint = torch.load(LOAD_PATH)
    selective_load(checkpoint['model_state_dict'], model)
    selective_load(checkpoint['optimizer_state_dict'], optimizer)
    for state in optimizer.state.values():
        for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        start_epoch = checkpoint['epoch']
        observations_count = checkpoint['observations_count']

    PATH = "model/LossModelData.tm"
    torch.save(
            {
                'epoch': start_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'observations_count': observations_count,
                'loss': loss_container
            }, PATH)
    
def plot_loss (x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(ylim=(0, 12))
    plt.xlabel('Integer Value')
    plt.ylabel('List Values')
    plt.title('Plot with Integer on X-axis and List on Y-axis')
    plt.grid(True)  # Add grid
    plt.savefig('LossGraph.png')

plot_loss(observations_container, loss_container)