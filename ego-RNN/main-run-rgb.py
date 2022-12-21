from __future__ import print_function, division

import os
import time

from objectAttentionModelConvLSTM import *
from load_data.spatial_transforms import (ToTensor, Normalize)
from load_data.video_spatial_transforms import (Compose, CenterCrop, Scale, MultiScaleCornerCrop, RandomHorizontalFlip)
from load_data.makeDataSet import makeDataset
import argparse
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main_run(dataset, stage, train_data_dir, val_data_dir, stage1_dict, out_dir, seqLen, trainBatchSize,
             valBatchSize, numEpochs, lr1, decay_factor, decay_step, memSize):

    if dataset == 'gtea61':
        num_classes = 61
    elif dataset == 'gtea71':
      num_classes = 71
    elif dataset == 'gtea_gaze':
        num_classes = 44
    elif dataset == 'egtea':
        num_classes = 106
    elif dataset == 'WJJ_label':
        num_classes = 12
    elif dataset == 'WJJ-label-1.1':
        num_classes = 15
    else:
        print('Dataset not found')
        sys.exit()

    model_folder = os.path.join('./', out_dir, dataset, 'stage'+str(stage))  # Dir for saving models and log files

    # Create the dir
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    else:
        print("The model_folder exists!")

    # Log files
    # writer = SummaryWriter(model_folder)
    train_log_loss = open((model_folder + '/train_log_loss.txt'), 'w')
    train_log_acc = open((model_folder + '/train_log_acc.txt'), 'w')
    val_log_loss = open((model_folder + '/val_log_loss.txt'), 'w')
    val_log_acc = open((model_folder + '/val_log_acc.txt'), 'w')


    # Data loader
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    pic_trans = Compose([ToTensor(), normalize])
    spatial_transform = {
        'vid': Compose([Scale(256), RandomHorizontalFlip(), MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224)]),
        'pic': pic_trans}

    vid_seq_train = makeDataset(train_data_dir,
                                spatial_transform=spatial_transform, seqLen=seqLen, fmt='.jpg')

    train_loader = torch.utils.data.DataLoader(vid_seq_train, batch_size=trainBatchSize,
                            shuffle=True, num_workers=4, pin_memory=True)
    if val_data_dir is not None:
        vid_seq_val = makeDataset(val_data_dir,
                                  spatial_transform={'vid': Compose([Scale(256), CenterCrop(224)]), 'pic': pic_trans},
                                  seqLen=seqLen, fmt='.jpg')

        val_loader = torch.utils.data.DataLoader(vid_seq_val, batch_size=valBatchSize,
                                shuffle=False, num_workers=2, pin_memory=True)
        valInstances = vid_seq_val.__len__()


    trainInstances = vid_seq_train.__len__()

    train_params = []
    if stage == 1:

        model = attentionModel(num_classes=num_classes, mem_size=memSize)
        model.train(False)
        for params in model.parameters():
            params.requires_grad = False
    else:

        model = attentionModel(num_classes=num_classes, mem_size=memSize)
        model.load_state_dict(torch.load(stage1_dict))
        model.train(False)
        for params in model.parameters():
            params.requires_grad = False
        #
        for params in model.resNet.layer4[0].conv1.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNet.layer4[0].conv2.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNet.layer4[1].conv1.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNet.layer4[1].conv2.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNet.layer4[2].conv1.parameters():
            params.requires_grad = True
            train_params += [params]
        #
        for params in model.resNet.layer4[2].conv2.parameters():
            params.requires_grad = True
            train_params += [params]
        #
        for params in model.resNet.fc.parameters():
            params.requires_grad = True
            train_params += [params]

        model.resNet.layer4[0].conv1.train(True)
        model.resNet.layer4[0].conv2.train(True)
        model.resNet.layer4[1].conv1.train(True)
        model.resNet.layer4[1].conv2.train(True)
        model.resNet.layer4[2].conv1.train(True)
        model.resNet.layer4[2].conv2.train(True)
        model.resNet.fc.train(True)

    for params in model.lstm_cell.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.classifier.parameters():
        params.requires_grad = True
        train_params += [params]


    model.lstm_cell.train(True)

    model.classifier.train(True)
    model.cuda()

    loss_fn = nn.CrossEntropyLoss()

    optimizer_fn = torch.optim.Adam(train_params, lr=lr1, weight_decay=4e-5, eps=1e-4)

    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_fn, milestones=decay_step,
                                                           gamma=decay_factor)

    train_iter = 0
    min_accuracy = 0
    start = time.time()
    for epoch in range(numEpochs):

        epoch_loss = 0
        numCorrTrain = 0
        trainSamples = 0
        iterPerEpoch = 0
        model.lstm_cell.train(True)
        model.classifier.train(True)
        # writer.add_scalar('lr', optimizer_fn.param_groups[0]['lr'], epoch+1)
        if stage == 2:
            model.resNet.layer4[0].conv1.train(True)
            model.resNet.layer4[0].conv2.train(True)
            model.resNet.layer4[1].conv1.train(True)
            model.resNet.layer4[1].conv2.train(True)
            model.resNet.layer4[2].conv1.train(True)
            model.resNet.layer4[2].conv2.train(True)
            model.resNet.fc.train(True)
        start_time = time.time()
        for i, (inputs, targets) in enumerate(train_loader):
            train_iter += 1
            iterPerEpoch += 1
            optimizer_fn.zero_grad()
            trainSamples += inputs.size(0)
            inputs = inputs.permute(2, 0, 1, 3, 4).cuda()
            label = targets.cuda()
            output_label, _ = model(inputs)
            loss = loss_fn(output_label, label)
            loss.backward()
            optimizer_fn.step()
            _, predicted = torch.max(output_label.data, 1)
            numCorrTrain += (predicted == label).sum()
            epoch_loss += loss.data
        avg_loss = epoch_loss/iterPerEpoch
        trainAccuracy = (numCorrTrain / trainSamples) * 100

        print('Train: Epoch = {} | Loss = {} | Accuracy = {}'.format(epoch+1, avg_loss, trainAccuracy), end='\t')
        end_time = time.time()
        print('Time: {} m'.format((end_time-start_time)/60))
        # writer.add_scalar('train/epoch_loss', avg_loss, epoch+1)
        # writer.add_scalar('train/accuracy', trainAccuracy, epoch+1)
        if val_data_dir is not None:
            if (epoch+1) % 5 == 0:
                model.train(False)
                val_loss_epoch = 0
                val_iter = 0
                val_samples = 0
                numCorr = 0
                with torch.no_grad():
                    for j, (inputs, targets) in enumerate(val_loader):
                        val_iter += 1
                        val_samples += inputs.size(0)
                        inputs = inputs.permute(2, 0, 1, 3, 4).cuda()
                        label = targets.cuda()
                        output_label, _ = model(inputs)
                        val_loss = loss_fn(output_label, label)
                        val_loss_epoch += val_loss.data
                        _, predicted = torch.max(output_label.data, 1)
                        numCorr += (predicted == label).sum()
                val_accuracy = (numCorr / val_samples) * 100
                avg_val_loss = val_loss_epoch / val_iter
                print('Val: Epoch = {} | Loss {} | Accuracy = {}'.format(epoch + 1, avg_val_loss, val_accuracy))
                # writer.add_scalar('val/epoch_loss', avg_val_loss, epoch + 1)
                # writer.add_scalar('val/accuracy', val_accuracy, epoch + 1)
                val_log_loss.write('Val Loss after {} epochs = {}\n'.format(epoch + 1, avg_val_loss))
                val_log_acc.write('Val Accuracy after {} epochs = {}%\n'.format(epoch + 1, val_accuracy))
                if val_accuracy > min_accuracy:
                    save_path_model = (model_folder + '/model_rgb_state_dict.pth')
                    torch.save(model.state_dict(), save_path_model)
                    min_accuracy = val_accuracy
            else:
                if (epoch+1) % 10 == 0:
                    save_path_model = (model_folder + '/model_rgb_state_dict_epoch' + str(epoch+1) + '.pth')
                    torch.save(model.state_dict(), save_path_model)
        optim_scheduler.step()
        print('\n')

    train_log_loss.close()
    train_log_acc.close()
    val_log_acc.close()
    val_log_loss.close()
    end = time.time()
    duration = end - start
    print('Time: {} m  {} h'.format(duration / 60, duration / 3600))
    # writer.export_scalars_to_json(model_folder + "/all_scalars.json")
    # writer.close()


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='WJJ-label-1.1', help='Dataset')
    parser.add_argument('--stage', type=int, default=2, help='Training stage')
    parser.add_argument('--trainDatasetDir', type=str, default=r'../read_txt/WJJ-label-1.1_train_read.txt',
                        help='Train set directory')
    parser.add_argument('--valDatasetDir', type=str, default=r'../read_txt/WJJ-label-1.1_test_read.txt',
                        help='Val set directory')
    parser.add_argument('--outDir', type=str, default='experiments', help='Directory to save results')
    parser.add_argument('--stage1Dict', type=str, default='./experiments/WJJ-label-1.1/stage1/model_rgb_state_dict.pth',
                        help='Stage 1 model path')
    parser.add_argument('--seqLen', type=int, default=8, help='Length of sequence')
    parser.add_argument('--trainBatchSize', type=int, default=32, help='Training batch size')
    parser.add_argument('--valBatchSize', type=int, default=16, help='Validation batch size')
    parser.add_argument('--numEpochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--stepSize', type=float, default=[25, 75], nargs="+", help='Learning rate decay step')
    parser.add_argument('--decayRate', type=float, default=0.1, help='Learning rate decay rate')
    parser.add_argument('--memSize', type=int, default=512, help='ConvLSTM hidden state size')

    args = parser.parse_args()

    dataset = args.dataset
    stage = args.stage
    trainDatasetDir = args.trainDatasetDir
    valDatasetDir = args.valDatasetDir
    outDir = args.outDir
    stage1Dict = args.stage1Dict
    seqLen = args.seqLen
    trainBatchSize = args.trainBatchSize
    valBatchSize = args.valBatchSize
    numEpochs = args.numEpochs
    lr1 = args.lr
    stepSize = args.stepSize
    decayRate = args.decayRate
    memSize = args.memSize

    main_run(dataset, stage, trainDatasetDir, valDatasetDir, stage1Dict, outDir, seqLen, trainBatchSize,
             valBatchSize, numEpochs, lr1, decayRate, stepSize, memSize)

__main__()
