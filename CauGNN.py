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
import datetime
import logging
import tempfile
from pathlib import Path
from ray import train
from ray.train import Checkpoint, get_checkpoint
import ray.cloudpickle as pickle

class CauGNN:
    def __init__(self, args: Namespace) -> None:
        # TODO: determine which initialisations can be done without A (done individually for each airline)
        self.args: Namespace = args
        self._set_savedir(args.modelID)
        self._data_loading()
        self._criterion_eval()
        self._plot_initialisation()
        self.metrics: dict = {'Training Loss': 0.0, 'RMSE': 0.0, 'RSE': 0.0, 'MAE': 0.0, 'RAE': 0.0, 'MSE':0.0, 'Correlation': 0.0}
        self.best_val: float = 10e15
        utils.model_logging(self.args, self.savedir)
        self._logsetup()
        self.A = None
        

        self.device = torch.device('cuda' if args.cuda else 'cpu')
 
        if not args.tune:
            self.start_epoch = 0

        if not args.airline_batching:
            config = None
            self._model_initialisation(config)

        

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

    def _model_initialisation(self,config) -> None:

        if not self.args.airline_batching: #Otherwise the TE matrix is loaded through Data.airline_matrix() in the run_airline_training function
            self._load_TE_matrix()

        if not self.args.n_e:
            self.n_e = self.A.shape[0]

        if self.args.tune:
            self.args.hid1 = config['hid1']
            self.args.hid2 = config['hid2']
            self.args.channel_size = config['channel_size']

        self.Model = TENet.Model(self.args, A=self.A)
        if self.args.cuda:
            self.Model.cuda()
            # if torch.cuda.device_count() > 1: #TODO: Test if this works to parallelize the model
            #     self.Model = nn.DataParallel(self.Model)
        self._optimiser(config)
        self.args.nParams = sum([p.nelement() for p in self.Model.parameters()])

        self.raytune_counter = 0
        if self.args.tune:
            checkpoint = get_checkpoint()
            if checkpoint:
                with checkpoint.as_directory() as checkpoint_dir:
                    data_path = Path(checkpoint_dir) / "data.pkl"
                    with open(data_path, "rb") as fp:
                        checkpoint_state = pickle.load(fp)
                    self.start_epoch = checkpoint_state["epoch"]
                    self.Model.load_state_dict(checkpoint_state["net_state_dict"])
                    self.optim.load_state_dict(checkpoint_state["optimizer_state_dict"])
            else:
                self.start_epoch = 0



    
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
        self.savedir = os.path.abspath(self.savedir)
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

    def _optimiser(self,config) -> None:

        if self.args.tune:
            self.args.lr = config['lr']


        # self.optim = Optim.Optim(
        #     self.Model.parameters(), self.args.optim, self.args.lr, self.args.clip,
        # )
        #! Use instead of Optim.Optim because of the Ray Tune integration
        self.optim = torch.optim.Adam(self.Model.parameters(), lr=self.args.lr)

    def _logsetup(self) -> None:
        logging.basicConfig(filename = os.path.join(self.savedir, 'train.log'), level=logging.INFO)
        self.logger = logging.getLogger('training')
        
    
    # TRAINING FUNCTIONS ------------------------------------------------------------------------------------------------
    def _training_pass(self, data: DataUtility) -> None:
        X: torch.Tensor = data.train[0]
        Y: torch.Tensor = data.train[1]
        self.Model.train()
        total_loss_training = 0
        n_samples = 0
        n_batches = 0

        for X, Y in data.get_batches(X, Y, self.args.batch_size, True):
            if X.shape[0] != self.args.batch_size:
                break
            self.Model.zero_grad()
            output = self.Model(X)
            #! How is he scaling when normalize = 1? --> He is scaling the output of the model by the maximum value of the entire matrix, when normalize = 1.
            #! However he does write the scaled values directly to the self.dat variable, used in the _batchfy function. The data.scale variable is not used and contains only ones.
            #! How is he scaling? --> He is scaling the output of the model by the maximum value of each column, when normalize = 2
            scale = data.scale.expand(output.size(0), data.cols).to(self.device)
            output_profit = output[:,-6] # Only the profit is used for the loss calculation
            Y_profit = Y[:,-6]
            scale_profit = scale[:,-6]
            loss = self.criterion(output_profit*scale_profit,Y_profit.to(self.device)*scale_profit) #Loss when only considering the profit
            # loss = self.criterion(output * scale, Y.to(self.device) * scale) #Loss when considering the whole feature set
            loss.backward()
            grad_norm = self.optim.step()
            total_loss_training += loss.data.item()
            # n_samples += (output.size(0) * data.cols) #Number of samples when considering the whole feature set
            n_samples += (output.size(0) * 1) #Number of samples when only considering the profit
            n_batches += 1
            torch.cuda.empty_cache()

        return total_loss_training / n_batches
    

    def run_epoch(self, data: DataUtility) -> None:

        print(f'----------- Starting epoch {self.epoch} ----------- \n \n')
        start_time = time.time()
        self.metrics['Training Loss'] = self._training_pass(data)
        self.train_loss_plot.append(self.metrics['Training Loss'])

        self.evaluate(data)

        if self.args.tune:
            checkpoint_data = {
                "epoch": self.raytune_counter,
                "net_state_dict": self.Model.state_dict(), #returns dict containing state of the model, which includes the model's parameters (stored in the model's layers)
                "optimizer_state_dict": self.optim.state_dict(),
            }
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "wb") as fp:
                    pickle.dump(checkpoint_data, fp, protocol=4) #protocol 5 only supported from Python 3.8

                checkpoint = Checkpoint.from_directory(checkpoint_dir)
                train.report(
                    {"Training Loss": self.metrics["Training Loss"], "Test MAE": self.metrics["MAE"]},
                    checkpoint=checkpoint,
                )
            self.raytune_counter += 1


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

        print(f'\n \n ----------- End of epoch {self.epoch}: Run Time = {round(end_time-start_time,2)}s ----------- \n \n ')
        

    def run_airline_training(self,config) -> None:        
        # First airline training
        airline = self.Data.airlines[0]
        self.A = self.Data.airline_matrix(airline)
        self._model_initialisation(config)
        self.run_training(self.Data.Airlines[airline])
    

        # Save model for first airline
        current_time = datetime.datetime.now().strftime("%Y-%m-%d")
        dir_path = os.path.join(self.savedir, f"AirlineID_{airline}_{current_time}")
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, 'model.pt'), 'wb') as f:
            torch.save(self.Model, f)


        # Subsequent airline training
        for airline in self.Data.airlines[1:]:
            self.A = self.Data.airline_matrix(airline)
            self.Model._set_A(self.A)
            self.run_training(self.Data.Airlines[airline])

            # Save model per airline
            current_time = datetime.datetime.now().strftime("%Y-%m-%d")
            dir_path = os.path.join(self.savedir, f"AirlineID_{airline}_{current_time}")
            os.makedirs(dir_path, exist_ok=True)
            with open(os.path.join(dir_path, 'model.pt'), 'wb') as f:
                torch.save(self.Model, f)
        
        vis.show_metrics(self.plot_metrics, run_name='CauGNN', save=self.savedir, vis=False)


    def run_training(self, data: DataUtility) -> None:

        print('Start Training -----------')
       
        for epoch in range(self.start_epoch+1, self.args.epochs + 1):
            if epoch == 1:  #! Right know best_val is not reset for every airline, because the model should only be saved if the validation loss is better than the best validation loss of all airlines
                self.best_val = 10e15
        
            self.epoch = epoch
            self.run_epoch(data)
            
        if not self.args.airline_batching:
            vis.show_metrics(self.plot_metrics, run_name='CauGNN', save=self.savedir, vis=False)


     # EVALUATION FUNCTIONS ----------------------------------------------------------------------------------------------   
    def evaluate(self, data: DataUtility):
        X: torch.Tensor = data.test[0]
        Y: torch.Tensor = data.test[1]
        
        self._eval_run(data, X, Y)   
        self._print_metrics() 
        self._plot_metrics() 
        
        
    def _eval_run(self, data: DataUtility, X: torch.Tensor, Y: torch.Tensor) -> None:
        self.Model.eval()
        total_MSE_test = 0
        total_MAE_test = 0
        total_RMSE_test = 0
        total_RAE_test = 0
        total_RSE_test = 0
        n_samples = 0
        n_batches = 0
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

                #Selecting the sixth last column uses only the profit for the loss calculation
                scale: torch.Tensor = data.scale.expand(output.size(0), data.cols).to(self.device)
                scale_profit = scale[:,-6] 
                output_profit = output[:,-6]
                output_scaled = output_profit * scale_profit
                Y_profit = Y[:,-6]
                Y_scaled = Y_profit.to(self.device) * scale_profit


                total_MSE_test += self.evaluateL2(output_scaled, Y_scaled).item() # L2 Loss equals MSE
                total_MAE_test += self.evaluateL1(output_scaled, Y_scaled).item() # L1 Loss equals MAE
                total_RMSE_test += math.sqrt(self.evaluateL2(output_scaled, Y_scaled).item()) 
                v1 = torch.sum(torch.abs(output_scaled - Y_scaled) / torch.sum(torch.abs(Y_scaled - torch.mean(Y_scaled)))).item()
                total_RAE_test += torch.sum(torch.abs(output_scaled - Y_scaled) / torch.sum(torch.abs(Y_scaled - torch.mean(Y_scaled)))).item() #RAE is the relative absolute error from the test set over the naive model, data.rae,(mean of the test set)
                total_RSE_test += torch.sum(torch.square(output_scaled - Y_scaled) / torch.sum(torch.square(Y_scaled - torch.mean(Y_scaled)))).item() #RSE is the relative squared error, same as RAE but squared
                n_samples += (output.size(0) * 1) #* x 1 only when predicting profit, when predicting the whole feature set then x data.cols
                n_batches += 1
                del scale, X, Y
                torch.cuda.empty_cache()

        self._calculate_metrics(data, total_MSE_test, total_MAE_test, total_RMSE_test, total_RAE_test, total_RSE_test, n_batches, predict, test)

    def _calculate_metrics(self, data: DataUtility, total_MSE_test, total_MAE_test, total_RMSE_test, total_RAE_test, total_RSE_test, n_batches, predict, test):
        self.metrics['RMSE'] = np.round(math.sqrt(total_RMSE_test / n_batches), 4)
        self.metrics['RSE'] = np.round(math.sqrt(total_RSE_test / n_batches), 4)
        self.metrics['RAE'] = np.round((total_RAE_test / n_batches), 4) 
        self.metrics['MAE'] = np.round(total_MAE_test / n_batches, 4)
        self.metrics['MSE'] = np.round(total_MSE_test / n_batches, 4)

        predict = predict.data.cpu().numpy()
        Ytest = test.data.cpu().numpy()
        sigma_p = (predict).std(axis=0)
        sigma_g = (Ytest).std(axis=0)
        mean_p = predict.mean(axis=0)
        mean_g = Ytest.mean(axis=0)
        index = (sigma_g != 0)
        correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
        self.metrics['Correlation'] = np.round((correlation[index]).mean(), 4)
        


    def _print_metrics(self) -> None:
        print(f'EPOCH {self.epoch} METRICS:')
        print(f'  Training Loss: {self.metrics["Training Loss"]} | MAE: {self.metrics["MAE"]} | MSE: {self.metrics["MSE"]} | RMSE: {self.metrics["RMSE"]} | RSE: {self.metrics["RSE"]} | RAE: {self.metrics["RAE"]} | Correlation: {self.metrics["Correlation"]}')
        self.logger.info(f'Epoch: {self.epoch} | Training Loss: {self.metrics["Training Loss"]} | MAE: {self.metrics["MAE"]} | MSE: {self.metrics["MSE"]} | RMSE: {self.metrics["RMSE"]} | RSE: {self.metrics["RSE"]} | RAE: {self.metrics["RAE"]} | Correlation: {self.metrics["Correlation"]}')


    def _plot_metrics(self) -> None:
        self.test_rmse_plot.append(self.metrics['RMSE'])
        self.test_mae_plot.append(self.metrics['MAE'])

        self.metric_mae = [self.train_loss_plot, self.test_mae_plot]
        self.metric_rmse = [[], self.test_rmse_plot]
        self.plot_metrics = {'MAE': self.metric_mae}

        if self.args.printc and self.epoch % 10 == 0 and self.epoch != self.args.epochs: # Call Function to Print Metrics and continuously update for each epoch
            for j, key in enumerate(self.plot_metrics.keys()):
                self.line1.set_ydata(self.plot_metrics[key][0]) # Update training line
                self.line1.set_xdata(list(range(0, self.epoch)))  
                self.line2.set_ydata(self.plot_metrics[key][1]) # Update test line
                self.line2.set_xdata(list(range(0, self.epoch)))  
            # Rescale the axes
            self.ax[0].relim()
            self.ax[0].autoscale_view()
            plt.draw()
            plt.pause(0.001)
        elif self.args.printc and self.epoch == self.args.epochs: # Call Function to keep the plot open after the last epoch
            for j, key in enumerate(self.plot_metrics.keys()):
                self.line1.set_ydata(self.plot_metrics[key][0]) # Update training line
                self.line1.set_xdata(list(range(0, self.epoch)))  
                self.line2.set_ydata(self.plot_metrics[key][1]) # Update test line
                self.line2.set_xdata(list(range(0, self.epoch)))  
            # Rescale the axes
            self.ax[0].relim()
            self.ax[0].autoscale_view()
            plt.ioff()
            plt.show()
            

    def _plot_initialisation(self):
        self.train_loss_plot: list = []
        self.test_rmse_plot: list = []
        self.test_mae_plot: list = []

        self.metric_mae = [self.train_loss_plot, self.test_mae_plot]
        self.metric_rmse = [[], self.test_rmse_plot]
        self.plot_metrics = {'MAE': self.metric_mae}

        # TODO: Not working with several airlines. Plot must be newly initialised for each airline
        if self.args.printc: # Call Function to Print Metrics
            self.fig, self.ax, self.line1, self.line2 = vis.show_metrics_continous(self.plot_metrics)
            plt.ion() # Turn on interactive mode