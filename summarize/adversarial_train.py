import torch
import numpy as np
import time
import copy
import torch.nn.functional as F
from summarize.attack import fgsm_attack
from summarize.normal_train import evaluate

def adv_train(model, train_loader, attack_loader, optimizer, scheduler, criterion, epsilon, epochs, device, save_model = False):
    '''
    attack loader:attack data loader, not validation data loader!!!!
    FGSM adversarial training
    hanyu's version, generate attack during training
    
    return: 
        train_loss_list
        test loss list
        train accuracy on original data
        train accuracy on self generating attack data
        model accuracy on attack data
    '''
    train_loss_list = []
    train_acc1_list = []
    val_acc_list = []
    val_loss_list = []
    train_acc2_list = []
    model.train()
    start_time = time.time()

    for epoch in range(epochs):
        total = 0.0
        train_acc1 = 0.0
        train_acc2 = 0.0
        train_loss = []

        model.train()
        for batch_num, (feats, labels) in enumerate(train_loader):

            feats, labels = feats.to(device), labels.to(device)

            # Set requires_grad attribute of tensor. Important for Attack
            feats.requires_grad = True

            optimizer.zero_grad()

            # calculate grad on origional image
            # with torch.cuda.amp.autocast():
            outputs1 = model(feats)
            loss1 = criterion(outputs1, labels.long())

            # create a copy to backward
            temp_feats = copy.deepcopy(feats)
            temp_model = copy.deepcopy(model)
            temp_outputs = temp_model(temp_feats)
            temp_loss = criterion(temp_outputs, labels.long())
            temp_model.zero_grad()
            temp_loss.backward()
            # generate adv image and calculate loss
            adv_ex = fgsm_attack(temp_feats, epsilon, temp_feats.grad.data, mask=torch.ones(feats.shape[0]).to(device))

            # with torch.cuda.amp.autocast():
            outputs2 = model(adv_ex)
            loss2 = criterion(outputs2, labels.long())

            # new loss,  update gradient loss on new loss
            loss = (loss1+loss2)/2
            loss.backward()
            
            optimizer.step()


            # calculate accuracy on original image
            prob1 = F.softmax(outputs1.detach(), dim=1)

            _, pred_labels1 = torch.max(prob1, 1)
            pred_labels1 = pred_labels1.view(-1)
            train_acc1 += torch.sum(torch.eq(pred_labels1, labels)).item()    

            # calculate accuracy on attack image
            prob2 = F.softmax(outputs2.detach(), dim=1)

            _, pred_labels2 = torch.max(prob2, 1)
            pred_labels2 = pred_labels2.view(-1)
            train_acc2 += torch.sum(torch.eq(pred_labels2, labels)).item()    
            
            total += len(labels)
            train_loss.extend([loss.item()]*feats.size()[0])
            torch.cuda.empty_cache()

            del loss, feats, labels, pred_labels1, pred_labels2
            del outputs1, outputs2, loss1, loss2
            del prob1, prob2, temp_feats, temp_loss, temp_outputs
            del temp_model, adv_ex
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
                        #'scheduler_state_dict' : scheduler.state_dict(),
            }, "Model_"+str(epoch))
    
        val_loss, val_acc = evaluate(model, attack_loader, criterion)
        print('epoch: %d\t'%(epoch+1),  'time: %d m: %d s '% divmod(time.time() - start_time, 60))
        start_time = time.time()
        train_loss_list.append(np.mean(train_loss))
        train_acc1_list.append(train_acc1/total)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        train_acc2_list.append(train_acc2/total)


        print('train_loss: %.5f\ttrain_acc1: %.5f' %(np.mean(train_loss), train_acc1/total))
        print('val_loss: %.5f\tval_acc: %.5f'% (val_loss, val_acc))
        print('attack_acc: %.5f'%(train_acc2/total))
        print('*'*70)
    return train_loss_list, val_loss_list, train_acc1_list, train_acc2_list, val_acc_list