import torch


def load_model(model, optimizer, scheduler, file_name):
    temp = torch.load(file_name) 
    model.load_state_dict(temp['model_state_dict'])
    optimizer.load_state_dict(temp['optimizer_state_dict'])
    scheduler.load_state_dict(temp['scheduler_state_dict'])
