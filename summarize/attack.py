import torch


def get_one_grad(model, criterion, data, target):
    # Set requires_grad attribute of tensor. Important for Attack
    data.requires_grad = True

    # Forward pass the data through the model
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

    # mask: 1 for correct, only update grad on correct image
    mask = torch.eq(init_pred.flatten(), target.flatten()).float()

    # Calculate the loss
    loss = criterion(output, target)

    # Zero all existing gradients
    model.zero_grad()

    # Calculate gradients of model in backward pass
    loss.backward()

    # Collect datagrad
    return data.grad.data, mask


def fgsm_attack(image, epsilon, data_grad, mask=None, model=None, criterion=None, target=None):
    '''
    image: batch x 3 x 32 x 32
    
    data_grad: batch x 3 x 32 x32
    
    mask: batch_size x 1 x 1 x 1, 1 for false prediction, 0 for correct prediction, use for accelarate computation
    '''
    if mask is None:
        sign_data_grad = data_grad.sign()
    else:
        # Collect the element-wise sign of the data gradient
        sign_data_grad = torch.mul(data_grad.sign(), mask.view(-1, 1, 1, 1))

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    # perturbed_image = torch.clamp(perturbed_image, -1, 1)
    # Return the perturbed image
    return perturbed_image


def mi_fgsm_attack(image, epsilon, data_grad, mask, model, criterion, target, decay_rate=1.0):
    rounds = 10.0
    alpha = epsilon / rounds
    grad = 0.0
    x = image.detach().clone()
    for t in range(int(rounds)):
        if t != 0:
            data_grad, mask = get_grad(model, criterion, x, target)
        grad = decay_rate * grad + data_grad / torch.norm(data_grad, p=1)
        if mask is None:
            sign_data_grad = data_grad.sign()
        else:
            # Collect the element-wise sign of the data gradient
            sign_data_grad = torch.mul(data_grad.sign(), mask.view(-1, 1, 1, 1))
        x = x + alpha * sign_data_grad
        x = x.detach().clone()
    return x
