
import torch
from model import Model
from torch.optim import Adam

def selective_load(source_dict, target):
    target.load_state_dict(source_dict)

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
    loss_holder = checkpoint['loss']

PATH = "model/ModelData.tm"
torch.save(
        {
            'epoch': start_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'observations_count': observations_count,
            'loss': loss_holder
        }, PATH)