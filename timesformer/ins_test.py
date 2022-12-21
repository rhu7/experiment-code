from __future__ import print_function, division
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
from timesformer.models.vit import TimeSformer
from load_data.spatial_transforms import (ToTensor, Normalize)
from load_data.video_spatial_transforms import Compose, CenterCrop, Scale
from load_data.makeDataSet import makeDataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import argparse


def main_run(dataset, model_state_dict, dataset_dir, seqLen, pretrained_weight):

    if dataset == 'gtea61':
        num_classes = 61
    elif dataset == 'WJJ-label-1.1':
        num_classes = 15

    log_folder = os.path.join('./TimeSformer_test_result', dataset)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    normalize = Normalize(mean=mean, std=std)
    spatial_transform = {'vid': Compose([Scale(224)]), 'pic': Compose([ToTensor(), normalize])}

    vid_seq_test = makeDataset(dataset_dir,
                               spatial_transform=spatial_transform,
                               seqLen=seqLen, fmt='.jpg', get_VidName=True)

    test_loader = torch.utils.data.DataLoader(vid_seq_test, batch_size=1,
                            shuffle=False, num_workers=2, pin_memory=True)

    model = TimeSformer(img_size=224, num_classes=num_classes, pretrained_model=pretrained_weight)
    model.load_state_dict(torch.load(model_state_dict))

    # for params in model.parameters():
    #     params.requires_grad = False

    model.train(False)
    model.cuda()
    test_samples = vid_seq_test.__len__()
    print('Number of samples = {}'.format(test_samples))
    print('Evaluating...')
    numCorr = 0
    true_labels = []
    predicted_labels = []
    pred, gt = np.zeros(num_classes), np.zeros(num_classes)
    error_info = []
    with torch.no_grad():
        for j, (inputs, targets, vid_name) in enumerate(test_loader):
            inputs = inputs.cuda()
            output_label = model(inputs)
            _, predicted = torch.max(output_label.data, 1)
            numCorr += (predicted == targets.cuda()).sum()
            p, g = int(predicted.item()), int(targets.item())
            if p == g:
                pred[p] += 1
            else:
                error_info.append((vid_name[0], p, g))
            gt[g] += 1
            true_labels.append(targets.numpy())
            predicted_labels.append(predicted.cpu().numpy())
    test_accuracy = (numCorr / test_samples) * 100
    every_acc = pred/gt
    print('Test Accuracy = {}%'.format(test_accuracy))
    print("Mean Class Acc:", every_acc.sum()/num_classes)
    print(every_acc)
    log_MCA = os.path.join(log_folder, 'TimeSformer_'+str(seqLen)+'_everyClassAcc.txt')
    log_TestAccuracy = os.path.join(log_folder, 'TimeSformer_'+str(seqLen)+'_Acc.txt')
    log_error_info = os.path.join(log_folder, 'TimeSformer_' + str(seqLen) + '_ErrorInfo.txt')
    with open(log_error_info, 'w') as fw:
        for info in error_info:
            print(info[0], info[1], info[2], file=fw)
    np.savetxt(log_MCA, every_acc)
    np.savetxt(log_TestAccuracy, test_accuracy.cpu().numpy()[None])


    # print(every_acc[27], every_acc[28])

    cnf_matrix = confusion_matrix(np.asarray(true_labels), np.asarray(predicted_labels)).astype(float)
    cnf_matrix_normalized = cnf_matrix / cnf_matrix.sum(axis=1)[:, np.newaxis]

    # ticks = np.linspace(0, 106, num=20)
    # plt.imshow(cnf_matrix_normalized, interpolation='none', cmap='binary')
    plt.matshow(cnf_matrix_normalized)
    plt.colorbar()
    # plt.xticks(ticks, fontsize=12)
    # plt.yticks(ticks, fontsize=12)
    plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 15})
    plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 15})
    plt.title(dataset)
    # plt.grid(True)
    # plt.clim(0, 1)
    save_path = os.path.join(log_folder, 'TimeSformer_'+str(seqLen)+'_cm.jpg')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='WJJ-label-1.1', help='Dataset')
    parser.add_argument('--datasetDir', type=str, default='../read_txt/WJJ-label-1.1_test_read.txt',
                        help='Dataset directory')
    parser.add_argument('--modelStateDict', type=str, default='./TimeSformer_experiments/WJJ-label-1.1/model_rgb_state_dict.pth',
                        help='Model path')
    parser.add_argument('--seqLen', type=int, default=8, help='Length of sequence')
    parser.add_argument('--pretrained', type=str, default='../pretrained_weight/TimeSformer_divST_8_224_SSv2.pyth',
                        help='The pretrained weight on other dataset')

    args = parser.parse_args()

    dataset = args.dataset
    model_state_dict = args.modelStateDict
    dataset_dir = args.datasetDir
    seqLen = args.seqLen
    pretrained = args.pretrained

    start = time.time()
    main_run(dataset, model_state_dict, dataset_dir, seqLen, pretrained)
    print('It takes {} minutes.'.format((time.time() - start)/60))

__main__()
