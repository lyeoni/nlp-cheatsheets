import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp

def load_distributed_model(path, distributed_model, rank, ext='.checkpoint'):
    path += ext
    
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    distributed_model.load_state_dict(torch.load(path, map_location=map_location))

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def show_first_layer_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
            break

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = SimpleModel().to(rank)
    distributed_model = DistributedDataParallel(model, device_ids=[rank])

    outputs = distributed_model(torch.randn(20, 10))
    
    # see model parameters : check model parameters
    show_first_layer_params(distributed_model)

    # load distributed model
    load_distributed_model('model', distributed_model, rank)

    # see model parameters : check model parameters after loading
    show_first_layer_params(distributed_model)

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)

if __name__=='__main__':
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        print(f"Requires at least 2 GPUs to run, but got {n_gpus}.")

    if not os.path.isfile('model.checkpoint'):
        print(f"Requires saved model checkpoint. To get model checkpoint, run save_distributed_model.py. first.")

    run_demo(demo_basic, 2)