import numpy as np
import torch
from torch.autograd import Variable
from DataUtility import DataUtility
from AirlineData import AirlineData
from typing import Union
import utils
import os
import data_utils
from argparse import Namespace
from TENet_master.models import TENet
from TENet_master.util import Teoriginal
import torch.nn as nn
import Optim
import vis
import matplotlib.pyplot as plt
import time
import math
import sys

class CauGNN:
    def __init__(self, args: Namespace) -> None:
        self.args: Namespace = args
        self._set_savedir(args.modelID)
        self._data_loading(args)
        self._load_TE_matrix(args)
        self.metrics: dict = {'Training Loss': 0.0, 'RMSE': 0.0, 'RSE': 0.0, 'MAE': 0.0, 'RAE': 0.0, 'Correlation': 0.0}
        self.best_val: float = 10e15

        self.Model = TENet.Model(args, self.A)
        if args.cuda:
            self.Model.cuda()
        self.nParams = sum([p.nelement() for p in self.Model.parameters()])
        print(f'Model has {self.nParams} parameters.')

    # INITIALISATION FUNCTIONS -------------------------------------------------------------------------------------------
    def _data_loading(self) -> None:
        assert self.args.data, '--data argument left empty. Please specify the location of the time series file.'
        if self.args.form41:
            rawdata = data_utils.form41_dataloader(self.args.data, self.args.airline_batching)
            if self.args.airline_batching:
                self.Data = AirlineData(rawdata, 0.8, self.args.cuda, self.args.horizon, self.args.window, self.args.normalize)
            else:
                self.Data = DataUtility(rawdata, 0.8, self.args.cuda, self.args.horizon, self.args.window, self.args.normalize)
        else: 
            rawdata = data_utils.dataloader(self.args.data)
            self.Data = DataUtility(rawdata, 0.8, self.args.cuda, self.args.horizon, self.args.window, self.args.normalize)
        
    def _load_TE_matrix(self):
        if not self.args.A:
            savepath = os.path.join(os.path.dirname(self.args.data), 'causality_matrix')
            if not os.path.isdir(savepath):
                os.makedirs(savepath)
            self.A = Teoriginal.calculate_te_matrix(self.args.data, savepath)
        elif self.args.A.endswith('.txt'):
            A = np.loadtxt(self.args.A)
            self.A = np.array(A, dtype=np.float32)
        self.n_e = self.A.shape[0] if not self.args.n_e else self.args.n_e

    def _set_savedir(self, modelID: str) -> None:
        if modelID:
            modelID = utils.set_modelID()
        self.modelID: str = modelID
        self.savedir: str = os.path.join('models', modelID)
        if not os.path.isdir(self.savedir):
            os.makedirs(self.savedir)   

    def _criterion_eval(self) -> None:
        if self.args.L1Loss:
            self.criterion = nn.L1Loss(size_average = False).cpu()
        else:
            self.criterion = nn.MSELoss(size_average = False).cpu()
        self.evaluateL1 = nn.L1Loss(size_average = False).cpu()
        self.evaluateL2 = nn.MSELoss(size_average = False).cpu()
        if self.args.cuda:
            self.criterion = self.criterion.cuda()
            self.evaluateL1 = self.evaluateL1.cuda()
            self.evaluateL2 = self.evaluateL2.cuda()

    def _optimiser(self) -> None:
        self.optim = Optim.Optim(
            self.model.parameters(), self.args.optim, self.args.lr, self.args.clip,
        )

    
    # TRAINING FUNCTIONS ------------------------------------------------------------------------------------------------
    def _training_pass(self, data: DataUtility) -> None:
        X: torch.Tensor = data.train[0]
        Y: torch.Tensor = data.train[1]
        self.model.train()
        total_loss = 0
        n_samples = 0

        for X, Y in data.get_batches(X, Y, self.args.batch_size, True):
            if X.shape[0] != self.args.batch_size:
                break
            self.model.zero_grad()
            output = self.model(X)
            scale = data.scale.expand(output.size(0), data.cols) # data.m = data.cols number of columns/nodes #? How is he scaling?
            loss = self.criterion(output * scale, Y * scale)
            loss.backward()
            grad_norm = self.optim.step()
            total_loss += loss.data.item()
            n_samples += (output.size(0) * data.cols)
            torch.cuda.empty_cache()

        return total_loss / n_samples
    

    def run_epoch(self, data: DataUtility) -> None:
        # TODO: epoch timing
        self.metrics['Training Loss'] = self._training_pass(data)
        self.train_loss_plot.append(self.metrics['Training Loss'])

        self.evaluate(data, mode='test')

        if str(self.metrics['Correlation']) == 'nan':
            sys.exit()

        if self.args.decoder == 'GIN':
            val = self.metrics['RSE']
        else:
            val = self.metrics['MAE']

        if val < self.best_val:
            with open(self.args.savepath, 'wb') as f:
                torch.save(self.model, f)
            self.best_val = val

        self.evaluate(data, mode='valid')
        

    def run_airline_training(self, data: AirlineData) -> None:
        for airline in data.airlines:
            # TODO: make sure model is saved and reloaded before training
            self.run_training(data.Data[airline])


    def run_training(self, data: DataUtility) -> None:
        print('Start Training -----------')
        #? should best_val be reset for every training run or left over the course of multiple training batches?  (multiple airlines)
        self.best_val = 10e15
        for epoch in range(1, self.args.epochs + 1):
            self.epoch = epoch
            self.run_epoch(data)


     # EVALUATION FUNCTIONS ----------------------------------------------------------------------------------------------   
    def evaluate(self, data: DataUtility, mode: str):
        self._plot_initialisation()
        if mode == 'test':
            X: torch.Tensor = data.test[0]
            Y: torch.Tensor = data.test[1]
        elif mode == 'valid':
            X: torch.Tensor = data.valid[0]
            Y: torch.Tensor = data.valid[1]
        
        self._eval_run(data, X, Y)   
        self._print_metrics(mode) 
        self._plot_metrics() 
        
        
    def _eval_run(self, data: DataUtility, X: torch.Tensor, Y: torch.Tensor) -> None:
        self.model.eval()
        total_loss = 0
        total_loss_l1 = 0
        n_samples = 0
        predict = None
        test = None

        with torch.no_grad():
            for X, Y in data.get_batches(X, Y, self.args.batch_size, False):
                if X.shape[0] != self.args.batch_size:
                    break
                output = self.model(X)

                if predict is None:
                    predict = output
                    test = Y
                else:
                    predict = torch.cat((predict, output))
                    test = torch.cat((test, Y))

                scale: torch.Tensor = data.scale.expand(output.size(0), data.cols)
                total_loss += self.evaluateL2(output * scale, Y * scale).item()
                total_loss_l1 += self.evaluateL1(output * scale, Y * scale).item()
                n_samples += (output.size(0) * data.cols)
                del scale, X, Y
                torch.cuda.empty_cache()

        self._calculate_metrics(total_loss, total_loss_l1, n_samples, predict, test)


    def _calculate_metrics(self, data: DataUtility, total_loss, total_loss_l1, n_samples, predict, test):
        self.metrics['RMSE'] = np.round(math.sqrt(total_loss / n_samples), 4)
        self.metrics['RSE'] = np.round(math.sqrt(total_loss / n_samples) / data.rse, 4)
        self.metrics['RAE'] = np.round((total_loss_l1 / n_samples) / data.rae, 4)
        predict = predict.data.cpu().numpy()
        Ytest = test.data.cpu().numpy()
        sigma_p = (predict).std(axis=0)
        sigma_g = (Ytest).std(axis=0)
        mean_p = predict.mean(axis=0)
        mean_g = Ytest.mean(axis=0)
        index = (sigma_g != 0)
        correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
        self.metrics['Correlation'] = np.round((correlation[index]).mean(), 4)
        self.metrics['MAE'] = np.round(total_loss_l1 / n_samples, 4)


    def _print_metrics(self, mode: str) -> None:
        if mode == 'test':
            print('END OF EPOCH METRICS:')
            print(f'  Training Loss: {self.metrics['Training Loss']} | RMSE: {self.metrics['RMSE']} | RSE: {self.metrics['RSE']} | MAE: {self.metrics['MAE']} | RAE: {self.metrics['RAE']} | Correlation: {self.metrics['Correlation']}')
        else: 
            print('VALIDATION METRICS:')
            print(f'  RMSE: {self.metrics['RMSE']} | RSE: {self.metrics['RSE']} | MAE: {self.metrics['MAE']} | RAE: {self.metrics['RAE']} | Correlation: {self.metrics['Correlation']}')


    def _plot_metrics(self) -> None:
        self.test_rmse_plot.append(self.metrics['RMSE'])
        self.test_mae_plot.append(self.metrics['MAE'])

        if self.args.print and self.epoch % 10 == 0: # Call Function to Print Metrics and continuously update for each epoch
            metric_rmse = [self.train_loss_plot, self.test_rmse_plot]
            metric_mae = [[], self.test_mae_plot]
            eval_metrics = {'RMSE': metric_rmse}
            for j, key in enumerate(eval_metrics.keys()):
                self.line1.set_ydata(eval_metrics[key][0]) # Update training line
                self.line1.set_xdata(list(range(0, self.epoch)))  
                self.line2.set_ydata(eval_metrics[key][1]) # Update test line
                self.line2.set_xdata(list(range(0, self.epoch)))  
            # Rescale the axes
            self.ax[0].relim()
            self.ax[0].autoscale_view()
            plt.draw()
            plt.pause(0.001)


    def _plot_initialisation(self):
        self.train_loss_plot = []
        self.test_rmse_plot = []
        self.test_mae_plot = []

        if self.args.print: # Call Function to Print Metrics
            self.metric_rmse = [self.train_loss_plot, self.test_rmse_plot]
            self.metric_mae = [[], self.test_mae_plot]
            self.eval_metrics = {'RMSE': self.metric_rmse}
            self.fig, self.ax, self.line1, self.line2 = vis.show_metrics_continous(self.eval_metrics)

