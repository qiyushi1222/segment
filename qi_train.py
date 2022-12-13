import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from new_model import Conv2dBidirRecurrentModel
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory
from qi_dataloader import model_dataset
from utils.fit_one_epoch import fit_one_epoch


if __name__ == "__main__":
    Cuda = False
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    num_classes = 2

    cls_weights     = np.ones([num_classes], np.float32)

    model_path = "model_data/ep092-loss0.044-val_loss0.054.pth"


    Freeze_Train = False

    Init_Epoch = 0
    Freeze_Epoch = 2
    Freeze_batch_size = 2

    UnFreeze_Epoch = 3
    Unfreeze_batch_size = 2

    Init_lr = 1e-1
    Min_lr = Init_lr * 0.01

    optimizer_type = "sgd"
    momentum = 0.9
    weight_decay = 0.8

    lr_decay_type = 'cos'

    save_period = 1

    save_dir = 'logs'

    VOCdevkit_path = 'VOCdevkit'

    dice_loss = True
    focal_loss = True

    num_workers = 1

    input_size = 64
    mode = 'GRU'
    hidden_size = 32
    kernel_size = (3, 3)
    num_layers = 1
    bias = True
    output_size = 65
    model = Conv2dBidirRecurrentModel(mode=mode,
                                      input_size=input_size,
                                      hidden_size=hidden_size,
                                      kernel_size=kernel_size,
                                      num_layers=num_layers,
                                      bias=bias,
                                      output_size=output_size).train()

    weights_init(model)


    if model_path != '':
        #------------------------------------------------------#
        #
        #------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        new_pre = {}
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model_dict = model.state_dict()
        pre_weights = torch.load(model_path, map_location=device)
        for k, v in pre_weights.items():  # 查看预训练网络参数各层叫什么名称
            k = 'rnn_cell_list.0.unet.' + k
            new_pre[k] = v

        # print(len(new_pre))

        del new_pre['rnn_cell_list.0.unet.resnet.conv1.weight']
        model.load_state_dict(new_pre, strict=False)


    input_shape = [320,320]
    loss_history = LossHistory(save_dir, model, input_shape=input_shape)
    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    num_train = 180
    num_val = 21

    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

    nbs = 16
    lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
    lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    optimizer = {
        'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    }[optimizer_type]

    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")


    train_dataset = model_dataset(mode='train')
    val_dataset = model_dataset(mode='val')

    gen = DataLoader(train_dataset, shuffle = False, batch_size=batch_size, num_workers=num_workers)
    gen_val = DataLoader(val_dataset,shuffle=False, batch_size=batch_size, num_workers=num_workers)

    for epoch in range(Init_Epoch, UnFreeze_Epoch):

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(model_train,model,loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,UnFreeze_Epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes, save_period, save_dir)

    loss_history.writer.close()


