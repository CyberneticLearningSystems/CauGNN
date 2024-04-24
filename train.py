import argparse
import math
import time
import Optim
import torch
import torch.nn as nn
import numpy as np
import importlib
import sys
import os
import pickle
import logging
import utils

import data_utils
from DataUtility import DataUtility
from AirlineData import AirlineData
# from ml_eval import *
from TENet_master.models import *
from TENet_master.util import Teoriginal
from eval import evaluate
np.seterr(divide='ignore',invalid='ignore')
from TENet_master.models import TENet
from vis import *


def training_pass(data: DataUtility, model: TENet.Model, criterion: str, optim: Optim.Optim, batch_size: int):
    X: torch.Tensor = data.train[0]
    Y: torch.Tensor = data.train[1]
    model.train()
    total_loss = 0
    n_samples = 0

    for X, Y in data.get_batches(X, Y, batch_size, True):
        if X.shape[0]!=args.batch_size:
            break
        model.zero_grad()
        output = model(X)
        scale = data.scale.expand(output.size(0), data.cols) # data.m = data.cols number of columns/nodes #? How is he scaling?
        loss = criterion(output * scale, Y * scale)
        loss.backward()
        grad_norm = optim.step()
        total_loss += loss.data.item()
        n_samples += (output.size(0) * data.cols)
        torch.cuda.empty_cache()

    return total_loss / n_samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multivariate Time series forecasting')
    parser.add_argument('--data', type=str, default=None, help='location of the data file')
    parser.add_argument('--n_e', type=int, default=None, help='The number of graph nodes')
    parser.add_argument('--model', type=str, default='TENet', help='Model type to use')
    parser.add_argument('--k_size', type=list, default=[3,5,7], help='number of CNN kernel sizes', nargs='*')
    parser.add_argument('--window', type=int, default=32, help='window size')
    parser.add_argument('--decoder', type=str, default= 'GNN', help = 'type of decoder layer')
    parser.add_argument('--horizon', type=int, default= 5, help = 'forecasting horizon')
    parser.add_argument('--A', type=str, default=None, help='Adjenency matrix, calculated from Transfer Entropy')
    parser.add_argument('--highway_window', type=int, default=1, help='The window size of the highway component')
    parser.add_argument('--channel_size', type=int, default=12, help='the channel size of the CNN layers')
    parser.add_argument('--hid1', type=int, default=40, help='the hidden size of the GNN layers')
    parser.add_argument('--hid2', type=int, default=10, help='the hidden size of the GNN layers')
    parser.add_argument('--clip', type=float, default=10, help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=54321, help='random seed')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=2000, metavar='N', help='report interval')
    parser.add_argument('--modelID', type=str,  default=None, help='model ID')
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--L1Loss', type=bool, default=True)
    parser.add_argument('--normalize', type=int, default=2)
    parser.add_argument('--output_fun', type=str, default=None)
    parser.add_argument('--num_adj', type=int, default=1)
    parser.add_argument('--attention_mode', type=str, default='naive')
    parser.add_argument('--skip_mode', type=str, default='concat')
    parser.add_argument('--form41', type=bool, default=False)
    parser.add_argument('--print', type=bool, default=False, help='prints the evaluation metric while training')
    parser.add_argument('--airline_batching', type=bool, default=False, help='Batch data by airline')
    args = parser.parse_args()

    # Data Loading
    assert args.data, '--data argument left empty. Please specify the location of the time series file.'
    if args.form41:
        rawdata = data_utils.form41_dataloader(args.data, args.airline_batching)
        if args.airline_batching:
            Data = AirlineData(rawdata, 0.6, 0.2, args.cuda, args.horizon, args.window, args.normalize)
        else:
            Data = DataUtility(rawdata, 0.6, 0.2, args.cuda, args.horizon, args.window, args.normalize)
    else: 
        rawdata = data_utils.dataloader(args.data)
        Data = DataUtility(rawdata, 0.6, 0.2, args.cuda, args.horizon, args.window, args.normalize)
    print(Data.rse)

    # Data, Adjacency Matrix and Nodes
    #? calculate TEmatrix for all airlines individually, or for the entire dataset?
    if not args.A:
        savepath = os.path.join(os.path.dirname(args.data), 'causality_matrix')
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        A = Teoriginal.calculate_te_matrix(args.data, savepath)
    elif args.A.endswith('.txt'):
        A = np.loadtxt(args.A)
        A = np.array(A, dtype=np.float32)
    if not args.n_e:
        args.n_e = A.shape[0]

    # Model ID and savepath definitions, logging setup
    if not args.modelID:
        args.modelID = utils.set_modelID()
    args.savepath = os.path.join('models', args.modelID)
    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath)    
    utils.logging_setup(args, __name__)
    
    # CUDA & seed settings
    if args.cuda:
        torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda True")
        else:
            torch.cuda.manual_seed(args.seed)

    # model = eval(args.model).Model(args,Data)
    model = TENet.Model(args, A)
    if args.cuda:
        model.cuda()

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)

    if args.L1Loss:
        criterion = nn.L1Loss(size_average = False).cpu()
    else:
        criterion = nn.MSELoss(size_average = False).cpu()
    evaluateL2 = nn.MSELoss(size_average = False).cpu()
    evaluateL1 = nn.L1Loss(size_average = False).cpu()
    if args.cuda:
        criterion = criterion.cuda()
        evaluateL1 = evaluateL1.cuda()
        evaluateL2 = evaluateL2.cuda()
        
        
    # Optim is a wrapper function to allow for the initialisation of multiple optimisers using one function in this scope
    optim = Optim.Optim(
        model.parameters(), args.optim, args.lr, args.clip,
    )

    try:
        print('begin training')
        train_loss_plot = []
        test_rmse_plot = []
        test_mae_plot = []

        if args.print: # Call Function to Print Metrics
            metric_rmse = [train_loss_plot, test_rmse_plot]
            metric_mae = [[], test_mae_plot]
            eval_metrics = {'RMSE': metric_rmse}
            fig, ax, line1, line2 = show_metrics_continous(eval_metrics)


        best_val = 10e15
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train_loss = training_pass(Data, model, criterion, optim, args.batch_size)
            train_loss_plot.append(train_loss)

            val_rmse, val_rse, val_mae, val_rae, val_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size)

            print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.5f} | valid rmse {:5.5f} |valid rse {:5.5f} | valid mae {:5.5f} | valid rae {:5.5f} |valid corr  {:5.5f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_rmse,val_rse, val_mae,val_rae, val_corr))
            # Save the model if the validation loss is the best we've seen so far.
            if str(val_corr) == 'nan':
                sys.exit()

            val = val_mae
            if args.decoder == 'GIN':
                val = val_rse

            if val < best_val:
                with open(args.savepath, 'wb') as f:
                    torch.save(model, f)
                best_val = val

                test_rmse, test_acc, test_mae,test_rae, test_corr  = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1, args.batch_size)
                print ("\n          test rmse {:5.5f} | test rse {:5.5f} | test mae {:5.5f} | test rae {:5.5f} |test corr {:5.5f}".format(test_rmse,test_acc, test_mae,test_rae, test_corr))
                test_rmse_plot.append(test_rmse)
                test_mae_plot.append(test_mae)
            else:
                test_rmse, test_acc, test_mae, test_rae, test_corr = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1, args.batch_size)
                print("\n          test rmse {:5.5f} |test rse {:5.5f} | test mae {:5.5f} | test rae {:5.5f} |test corr {:5.5f}".format(test_rmse, test_acc, test_mae, test_rae, test_corr))
                test_rmse_plot.append(test_rmse)
                test_mae_plot.append(test_mae)
            

            if args.print and epoch % 10 == 0: # Call Function to Print Metrics and continuously update for each epoch
                metric_rmse = [train_loss_plot, test_rmse_plot]
                metric_mae = [[], test_mae_plot]
                eval_metrics = {'RMSE': metric_rmse}
                for j, key in enumerate(eval_metrics.keys()):
                    line1.set_ydata(eval_metrics[key][0]) # Update training line
                    line1.set_xdata(list(range(0,epoch)))  
                    line2.set_ydata(eval_metrics[key][1]) # Update test line
                    line2.set_xdata(list(range(0,epoch)))  
                # Rescale the axes
                ax[0].relim()
                ax[0].autoscale_view()
                plt.draw()
                plt.pause(0.001)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')



    # Load the best saved model.
    with open(args.savepath, 'rb') as f:
        model = torch.load(f)
    test_mse, test_acc, test_mae,test_rae, test_corr  = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1, args.batch_size)
    print ("\ntest rmse {:5.5f} |test rse {:5.5f} | test mae {:5.5f} | test rae {:5.5f} |test corr {:5.5f}".format(test_mse,test_acc, test_mae,test_rae, test_corr))


    # Print Metrics
    models = ['model.pt']
    run_name = 'Form41_quarterly'
    metric_rmse = [train_loss_plot, test_rmse_plot]
    metric_mae = [[], test_mae_plot]
    eval_metrics = {'RMSE': metric_rmse, 'MAE': metric_mae}
    #Save evaluation metric as file
    with open(os.path.join(args.savepath, 'eval_dat.json'), 'wb') as f:
        pickle.dump(eval_metrics, f)
    
    fig2 = show_metrics(models, eval_metrics, args.modelID, vis=False, save=args.savepath)

    