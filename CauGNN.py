import numpy as np
import torch
from DataUtility import DataUtility
from AirlineData import AirlineData
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
import math
import sys
import time
import logging

class CauGNN:
    def __init__(self, args: Namespace) -> None:
        # TODO: determine which initialisations can be done without A (done individually for each airline)
        self.args: Namespace = args
        self._set_savedir(args.modelID)
        self._data_loading()
        self._criterion_eval()
        self.metrics: dict = {'Training Loss': 0.0, 'RMSE': 0.0, 'RSE': 0.0, 'MAE': 0.0, 'RAE': 0.0, 'Correlation': 0.0}
        self.best_val: float = 10e15
        utils.model_logging(self.args, self.savedir)
        self._logsetup()
        self.A = None

        self.device = torch.device('cuda' if args.cuda else 'cpu')

        if not args.airline_batching:
            self._model_initialisation()
        

    # INITIALISATION FUNCTIONS -------------------------------------------------------------------------------------------
    def _data_loading(self) -> None:
        assert self.args.data, '--data argument left empty. Please specify the location of the time series file.'
        if self.args.form41:
            rawdata = data_utils.form41_dataloader(self.args.data, self.args.airline_batching)
            if self.args.airline_batching:
                self.Data = AirlineData(self.args, 0.8, rawdata)
            else:
                self.Data = DataUtility(self.args, 0.8, rawdata)
        else: 
            rawdata = data_utils.dataloader(self.args.data)
            self.Data = DataUtility(self.args, 0.8, rawdata)

    def _model_initialisation(self) -> None:

        self._load_TE_matrix()
        if not self.args.n_e:
            self.n_e = self.A.shape[0]
        self.Model = TENet.Model(self.args, A=self.A)
        if self.args.cuda:
            self.Model.cuda()
        self._optimiser()
        self.args.nParams = sum([p.nelement() for p in self.Model.parameters()])
        
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
        if not modelID:
            modelID = utils.set_modelID()
        self.modelID: str = modelID
        self.savedir: str = os.path.join('models', self.modelID)
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
            self.Model.parameters(), self.args.optim, self.args.lr, self.args.clip,
        )

    def _logsetup(self) -> None:
        logging.basicConfig(filename = os.path.join(self.savedir, 'train.log'), level=logging.INFO)
        self.logger = logging.getLogger('training')
        
    
    # TRAINING FUNCTIONS ------------------------------------------------------------------------------------------------
    def _training_pass(self, data: DataUtility) -> None:
        X: torch.Tensor = data.train[0]
        Y: torch.Tensor = data.train[1]
        self.Model.train()
        total_loss = 0
        n_samples = 0

        for X, Y in data.get_batches(X, Y, self.args.batch_size, True):
            if X.shape[0] != self.args.batch_size:
                break
            self.Model.zero_grad()
            output = self.Model(X)
            #! How is he scaling when normalize = 1? --> He is scaling the output of the model by the maximum value of the entire matrix, when normalize = 1.
            #! However he does writes the scaled values directly to the self.dat variable, used in the _batchfy function. The data.scale variable is not used and contains only ones.
            #! How is he scaling? --> He is scaling the output of the model by the maximum value of each row, when normalize = 2
            scale = data.scale.expand(output.size(0), data.cols).to(self.device) # data.m = data.cols number of columns/nodes #? How is he scaling?
            loss = self.criterion(output * scale, Y.to(self.device) * scale)
            loss.backward()
            grad_norm = self.optim.step()
            total_loss += loss.data.item()
            n_samples += (output.size(0) * data.cols)
            torch.cuda.empty_cache()

        return total_loss / n_samples
    

    def run_epoch(self, data: DataUtility) -> None:
        # TODO: epoch timing
        print(f'Starting epoch {self.epoch} -----------')
        start_time = time.time()
        self._plot_initialisation()
        self.metrics['Training Loss'] = self._training_pass(data)
        self.train_loss_plot.append(self.metrics['Training Loss'])

        self.evaluate(data)

        if str(self.metrics['Correlation']) == 'nan':
            sys.exit()

        if self.args.decoder == 'GIN':
            val = self.metrics['RSE']
        else:
            val = self.metrics['MAE']

        if val < self.best_val:
            with open(os.path.join(self.savedir, 'model.pt'), 'wb') as f:
                torch.save(self.Model, f)
            self.best_val = val
        end_time = time.time()
        print(f'End of epoch {self.epoch}: Run Time = {round(end_time-start_time,2)}s -----------')
        

    def run_airline_training(self) -> None:
        # First airline training
        airline = self.Data.airlines[0]
        self.A = self.Data.airline_matrix(airline)
        self._model_initialisation()
        self.run_training(self.Data.Airlines[airline])

        # Subsequent airline training
        for airline in self.Data.airlines[1:]:
            self.A = self.Data.airline_matrix(airline)
            # TODO: make sure model is saved and reloaded before training
            self.Model._set_A(self.A)
            self.run_training(self.Data.Airlines[airline])


    def run_training(self, data: DataUtility) -> None:
        print('Start Training -----------')
        #? should best_val be reset for every training run or left over the course of multiple training batches?  (multiple airlines)
        self.best_val = 10e15
        for epoch in range(1, self.args.epochs + 1):
            self.epoch = epoch
            self.run_epoch(data)


     # EVALUATION FUNCTIONS ----------------------------------------------------------------------------------------------   
    def evaluate(self, data: DataUtility):
        # self._plot_initialisation() #! This is not needed here, as it is already called in the run_epoch function
        X: torch.Tensor = data.test[0]
        Y: torch.Tensor = data.test[1]
        
        self._eval_run(data, X, Y)   
        self._print_metrics() 
        self._plot_metrics() 
        
        
    def _eval_run(self, data: DataUtility, X: torch.Tensor, Y: torch.Tensor) -> None:
        self.Model.eval()
        total_loss = 0
        total_loss_l1 = 0
        n_samples = 0
        predict = None
        test = None

        with torch.no_grad():
            for X, Y in data.get_batches(X, Y, self.args.batch_size, False):
                if X.shape[0] != self.args.batch_size:
                    break
                output = self.Model(X)

                if predict is None:
                    predict = output
                    test = Y
                else: #? Why is here the output and Y tensors concatenated with test and predict?
                    predict = torch.cat((predict, output))
                    test = torch.cat((test, Y))

                scale: torch.Tensor = data.scale.expand(output.size(0), data.cols).to(self.device)
                total_loss += self.evaluateL2(output * scale, Y.to(self.device) * scale).item()
                total_loss_l1 += self.evaluateL1(output * scale, Y.to(self.device) * scale).item()
                n_samples += (output.size(0) * data.cols)
                del scale, X, Y
                torch.cuda.empty_cache()

        self._calculate_metrics(data, total_loss, total_loss_l1, n_samples, predict, test)

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


    def _print_metrics(self) -> None:
        print(f'EPOCH {self.epoch} METRICS:')
        print(f'  Training Loss: {self.metrics["Training Loss"]} | RMSE: {self.metrics["RMSE"]} | RSE: {self.metrics["RSE"]} | MAE: {self.metrics["MAE"]} | RAE: {self.metrics["RAE"]} | Correlation: {self.metrics["Correlation"]}')
        self.logger.info(f'Epoch: {self.epoch} | Training Loss: {self.metrics["Training Loss"]} | RMSE: {self.metrics["RMSE"]} | RSE: {self.metrics["RSE"]} | MAE: {self.metrics["MAE"]} | RAE: {self.metrics["RAE"]} | Correlation: {self.metrics["Correlation"]}')


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
