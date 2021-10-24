from os import device_encoding
import torch
import numpy as np
import torch.nn.functional as F
import time

def evaluate(model, test_loader, criterion, device, return_pred=False):
    '''
    pred(np.array): if True, return prediction label, defalt False
    '''
    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        correct_predictions = 0.0
        pred = []

        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)

            output = model(x)

            _, predicted = torch.max(output.data, 1)
            correct_predictions += (predicted == y).sum().item()

            loss = criterion(output, y).detach()
            running_loss += loss.item()

            if return_pred:
                pred.append(predicted.detach().cpu().numpy())

        total = len(test_loader.dataset)
        running_loss /= total
        acc = (correct_predictions/total)*100.0
        # print('Testing Loss: ', running_loss)
        # print('Testing Accuracy: ', acc, '%')
        if return_pred:
            pred = np.concatenate(pred)
            return running_loss, acc, pred
        else:
            return running_loss, acc


def train(model, train_loader, test_loader, optimizer, criterion, epochs, device, scheduler = None, save_model = False):
    '''
    return None
    
    loss will print during training
    
    save_model: save optimizer and model scheduler
    '''
    model.train()
    start_time = time.time()

    for epoch in range(epochs):
        total = 0.0
        train_acc = 0.0
        train_loss = []

        model.train()
        for batch_num, (feats, labels) in enumerate(train_loader):
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(feats)
                loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            prob = F.softmax(outputs.detach(), dim=1)

            _, pred_labels = torch.max(prob, 1)
            pred_labels = pred_labels.view(-1)
            train_acc += torch.sum(torch.eq(pred_labels, labels)).item()    
            
            total += len(labels)
            train_loss.extend([loss.item()]*feats.size()[0])
            torch.cuda.empty_cache()

            del loss, feats, labels, pred_labels, outputs, prob
        
        if scheduler is not None:
            scheduler.step()
        if save_model and scheduler is not None:
            torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict' : scheduler.state_dict(),
            }, "Model_"+str(epoch))
        if save_model and scheduler is None:
            torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
            }, "Model_"+str(epoch))
        
            
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        
        print('epoch: %d\t'%(epoch+1),  'time: %d m: %d s '% divmod(time.time() - start_time, 60))
        start_time = time.time()
        print('train_loss: %.5f\ttrain_acc: %.5f' %(np.mean(train_loss), train_acc/total))
        print('val_loss: %.5f\tval_acc: %.5f'% (val_loss, val_acc))
        print('*'*60)