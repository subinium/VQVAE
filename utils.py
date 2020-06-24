import torch

def load_model(model, optimizer, iteration, checkpoint_path):
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    model.load_state_dict(checkpoint_dict['model'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    return model, optimizer, iteration

def save_model(model, optimizer, iteration, checkpoint_path):
    print(f'Save model and optimizer state at itertation {iteration} to {checkpoint_path}')
    torch.save({
        'iteration' : iteration,
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict()
    }, checkpoint_path)
    