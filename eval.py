import argparse
import math
import time
import Optim
import torch
import torch.nn as nn
import numpy as np
import importlib
import sys
from data_utils import *
from ml_eval import *
from TENet_master.models import TENet
from DataUtility import DataUtility

np.seterr(divide='ignore', invalid='ignore')


def evaluate(data, X: torch.Tensor, Y: torch.Tensor, model: TENet, evaluateL2: nn.MSELoss, evaluateL1: nn.L1Loss, batch_size: int):
    # model.eval() just sets the model to evaluation mode, which turns off certain modules (like dropout and batch normalisation)
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    with torch.no_grad():
        for X, Y in data.get_batches(X, Y, batch_size, False):
            if X.shape[0] != batch_size:
                break
            output = model(X)

            if predict is None:
                predict = output
                test = Y
            else:
                predict = torch.cat((predict, output))
                test = torch.cat((test, Y))

            scale = data.scale.expand(output.size(0), data.cols)
            total_loss += evaluateL2(output * scale, Y * scale).item()
            total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
            n_samples += (output.size(0) * data.cols)
            del scale, X, Y
            torch.cuda.empty_cache()

    rmse = math.sqrt(total_loss / n_samples)
    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae
    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    mae = total_loss_l1 / n_samples

    return rmse, rse, mae, rae, correlation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multivariate Time series forecasting')
    parser.add_argument('--data', type=str,
                        default="/home/jiangnanyida/Documents/MTS/MTS_TEGNN/TENet-master/data/exchange_rate.txt",
                        help='location of the data file')
    parser.add_argument('--window', type=int, default=32, help='window size')
    parser.add_argument('--decoder', type=str, default='GNN', help='type of decoder layer')
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save', type=str, default='model/model.pt', help='path to save the final model')
    parser.add_argument('--cuda', type=bool, default=True) #! In cmd line --cuda "" must be added to make it false, any other value will be true
    parser.add_argument('--normalize', type=int, default=2)
    parser.add_argument('--L1Loss', type=bool, default=True)

    args = parser.parse_args()

    Data = DataUtility(args.data, 0.8, args.cuda, args.horizon, args.window, args.normalize)

    if args.L1Loss:
        criterion = nn.L1Loss(size_average=False).cpu()
    else:
        criterion = nn.MSELoss(size_average=False).cpu()
    evaluateL2 = nn.MSELoss(size_average=False).cpu()
    evaluateL1 = nn.L1Loss(size_average=False).cpu()
    if args.cuda:
        criterion = criterion.cuda()
        evaluateL1 = evaluateL1.cuda()
        evaluateL2 = evaluateL2.cuda()

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)
    test_mse, test_acc, test_mae, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2,
                                                                evaluateL1, args.batch_size)
    print("\ntest rmse {:5.5f} |test rse {:5.5f} | test mae {:5.5f} | test rae {:5.5f} |test corr {:5.5f}".format(test_mse,
                                                                                                                test_acc,
                                                                                                                test_mae,
                                                                                                                test_rae,
                                                                                                                test_corr))
    