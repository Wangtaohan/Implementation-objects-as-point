
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.centernet import CenterNet_HourglassNet, CenterNet_Resnet50
from utils.callbacks import LossHistory
from utils.dataloader import CenternetDataset, centernet_dataset_collate
from utils.utils import get_classes
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    Cuda            = True
    classes_path    = 'model_data/voc_classes.txt'
    model_path      = 'model_data/centernet_resnet50_voc.pth'
    input_shape     = [512, 512]
    backbone        = "resnet50"
    pretrained      = False
    
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 16
    Freeze_lr           = 1e-3

    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 8
    Unfreeze_lr         = 1e-4
    
    Freeze_Train        = True
    num_workers         = 4
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'
    class_names, num_classes = get_classes(classes_path)

    if backbone == "resnet50":
        model = CenterNet_Resnet50(num_classes, pretrained = pretrained)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    loss_history    = LossHistory("logs/")
    
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if True:
        batch_size  = Freeze_batch_size
        lr          = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch   = Freeze_Epoch
                        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("dataset too small")
        
        optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset   = CenternetDataset(train_lines, input_shape, num_classes, train = True)
        val_dataset     = CenternetDataset(val_lines, input_shape, num_classes, train = False)
        gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=centernet_dataset_collate)
        gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=centernet_dataset_collate)

        if Freeze_Train:
            model.freeze_backbone()

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, loss_history, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda, backbone)
            lr_scheduler.step()
            
    if True:
        batch_size  = Unfreeze_batch_size
        lr          = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch   = UnFreeze_Epoch
                        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("dataset too small")
        
        optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset   = CenternetDataset(train_lines, input_shape, num_classes, train = True)
        val_dataset     = CenternetDataset(val_lines, input_shape, num_classes, train = False)
        gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=centernet_dataset_collate)
        gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=centernet_dataset_collate)

        if Freeze_Train:
            model.unfreeze_backbone()
                
        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, loss_history, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda, backbone)
            lr_scheduler.step()
