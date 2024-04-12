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

from utils import *
from ml_eval import *
from TENet_master.models import *
from TENet_master.util import Teoriginal
from eval import evaluate
np.seterr(divide='ignore',invalid='ignore')
from TENet_master.models import TENet

def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0

    for X, Y in data.get_batches(X, Y, batch_size, True):
        if X.shape[0]!=args.batch_size:
            break
        model.zero_grad()
        output = model(X)
        scale = data.scale.expand(output.size(0), data.m) #data.m number of columns/nodes #? How is he scaling?
        loss = criterion(output * scale, Y * scale)
        loss.backward()
        grad_norm = optim.step()
        total_loss += loss.data.item()
        n_samples += (output.size(0) * data.m)
        torch.cuda.empty_cache()

    return total_loss / n_samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multivariate Time series forecasting')
    parser.add_argument('--data', type=str, default=None, help='location of the data file')
    parser.add_argument('--n_e', type=int, default=8, help='The number of graph nodes')
    parser.add_argument('--model', type=str, default='TENet', help='Model type to use')
    parser.add_argument('--k_size', type=list, default=[3,5,7], help='number of CNN kernel sizes', nargs='*')
    parser.add_argument('--window', type=int, default=32, help='window size')
    parser.add_argument('--decoder', type=str, default= 'GNN', help = 'type of decoder layer')
    parser.add_argument('--horizon', type=int, default= 5)
    parser.add_argument('--A', type=str, default=None, help='A')
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
    parser.add_argument('--save', type=str,  default='model/model.pt', help='path to save the final model')
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
    args = parser.parse_args()

    assert args.data, '--data arg left empty. Please specify the location of the time series file.'
    if not args.A:
        savepath = os.path.join(os.path.dirname(args.data), 'causality_matrix')
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        args.A = Teoriginal.calculate_te_matrix(args.data, savepath)
    elif args.A.endswith('.txt'):
        A = np.loadtxt(args.A)
        A = np.array(A, dtype=np.float32)
    if not args.n_e:
        args.n_e = args.A.shape[0]

    if not os.path.isdir(os.path.dirname(args.save)):
        os.makedirs(os.path.dirname(args.save))
    
    # args.cuda = args.gpu is not None
    # args.cuda = False
    if args.cuda:
        torch.cuda.set_device(args.gpu)
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda True")
        else:
            torch.cuda.manual_seed(args.seed)

    Data = Data_utility(args.data, 0.6, 0.2, args.cuda, args.horizon, args.window, args.normalize, form41=args.form41)
    print(Data.rse)

    # model = eval(args.model).Model(args,Data)
    model = TENet.Model(args,Data)
    #
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
        
        
    best_val = 111110
    optim = Optim.Optim(
        model.parameters(), args.optim, args.lr, args.clip,
    )
    # Optim is a wrapper function to allow for the initialisation of multiple optimisers using one function in this scope

    try:
        print('begin training')
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)

            val_rmse, val_rse, val_mae, val_rae, val_corr = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1, args.batch_size)

            print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.5f} | valid rmse {:5.5f} |valid rse {:5.5f} | valid mae {:5.5f} | valid rae {:5.5f} |valid corr  {:5.5f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_rmse,val_rse, val_mae,val_rae, val_corr))
            # Save the model if the validation loss is the best we've seen so far.
            if str(val_corr) == 'nan':
                sys.exit()

            val = val_mae
            if args.decoder == 'GIN':
                val = val_rse

            if val < best_val:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val = val

                test_rmse, test_acc, test_mae,test_rae, test_corr  = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size)
                print ("\n          test rmse {:5.5f} |test rse {:5.5f} | test mae {:5.5f} | test rae {:5.5f} |test corr {:5.5f}".format(test_rmse,test_acc, test_mae,test_rae, test_corr))
            else:
                test_rmse, test_acc, test_mae, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size)
                print("\n          test rmse {:5.5f} |test rse {:5.5f} | test mae {:5.5f} | test rae {:5.5f} |test corr {:5.5f}".format(test_rmse, test_acc, test_mae, test_rae, test_corr))

            # if epoch % 5 == 0:
            #     test_rmse,test_acc, test_mae,test_rae, test_corr  = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size)
            #     print ("\ntest rmse {:5.5f} |test rse {:5.5f} | test mae {:5.5f} | test rae {:5.5f} |test corr {:5.5f}".format(test_rmse,test_acc, test_mae,test_rae, test_corr))

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)
    test_mse,test_acc, test_mae,test_rae, test_corr  = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size)
    print ("\ntest rmse {:5.5f} |test rse {:5.5f} | test mae {:5.5f} | test rae {:5.5f} |test corr {:5.5f}".format(test_mse,test_acc, test_mae,test_rae, test_corr))
    