import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import argparse
from typing import List
import math
import os
import utils
import shutil



from AirlineData import AirlineData
from DataUtility import DataUtility
import data_utils
import vis

class ModelTest:
    def __init__(self, args):
        self.args = args
        self.batch_size: int = 1
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.cuda else 'cpu')
        self._set_savedir(self.args.modelID)

        self._model_loading(self.args.modelID)
        self._data_loading(self.args.data,self.args.airlineID)
        self._normalized(self.rawdata)

        self.metrics: dict = {'RMSE': 0.0, 'RSE': 0.0, 'MAE': 0.0, 'RAE': 0.0, 'MSE':0.0, 'Correlation': 0.0}
        self._criterion_eval()
        self._plot_initialisation()

        eval_set = range(self.args.window+self.args.horizon-1, self.rows)
        self.eval_data = self._batchify(eval_set)

        self.scale: torch.Tensor = torch.from_numpy(self.scale).float()

        if self.args.cuda:
            self.scale = self.scale.cuda()
        self.scale = Variable(self.scale)       

    def _data_loading(self, data_path: str, airlineID: str) -> None:

        rawdata = data_utils.load_form41(data_path)
        rawdata.dropna(inplace=True, axis=0)
        rawdata = rawdata[rawdata['AIRLINE_ID'] == airlineID]
        self.timescale = rawdata['YEAR'].astype(str) + '-' + rawdata['QUARTER'].astype(str)
        try:#If the columns are already dropped
            rawdata.drop(columns=['UNIQUE_CARRIER_NAME', 'YEAR', 'QUARTER'], inplace=True)
        except KeyError: 
            pass
        
        self.rawdata = rawdata


    def _normalized(self, rawdat: pd.DataFrame):

        if (self.args.normalize == 3): # normalized by the maximum value of each col(sensor).
            data_to_scale = rawdat.drop(columns=['AIRLINE_ID']).copy()
            self.cols = data_to_scale.shape[1] 
            self.rows = data_to_scale.shape[0]
            self.scale: np.ndarray[float] = np.ones(self.cols)
        
            self.data_scaled = data_to_scale.copy()
            for i in range(self.cols):
                self.scale[i] = np.max(np.abs(data_to_scale.iloc[:,i]))
                if self.scale[i] == 0:
                    self.data_scaled.iloc[:,i] = self.data_scaled.iloc[:,i]
                else:
                    self.data_scaled.iloc[:,i] = self.data_scaled.iloc[:,i] / self.scale[i]
        
            self.data_np: np.ndarray[float] = np.array(self.data_scaled, dtype=float)
            

        elif (self.args.normalize == 4): # normalizes only the profit column as the rest is normalized through the pca in advance
            data_to_scale = rawdat.drop(columns=['AIRLINE_ID']).copy()
            self.cols = data_to_scale.shape[1]
            self.rows = data_to_scale.shape[0]
            self.scale: np.ndarray[float] = np.ones(self.cols)
            self.data_scaled = data_to_scale.copy()

            data_to_scale = data_to_scale.iloc[:,-1]
            self.scale[-1] = np.max(np.abs(data_to_scale))

            if self.scale[-1] == 0:
                self.data_scaled.iloc[:,-1] = self.data_scaled.iloc[:,-1]
            else:
                self.data_scaled.iloc[:,-1] = self.data_scaled.iloc[:,-1] / self.scale[-1]
            
            self.data_np: np.ndarray[float] = np.array(self.data_scaled, dtype=float)
  
        else:
            raise NotImplementedError('Normalization not set to 3 or 4 in arguments')
        

    def _batchify(self, idx_set: int) -> List[torch.Tensor]:
        '''
        Generates the feature and target vectors for the model by considering the length of the training and test set. The feature vector is a tensor of shape (n, window, cols) and the target vector is a tensor of shape (n, cols).
        The outcome of _batchify() is used in get_batches to generate a batch of samples from each set (train and test).
        '''
        
        n: int = len(idx_set)
        X: torch.Tensor = torch.zeros((n,self.args.window,self.cols))
        Y: torch.Tensor = torch.zeros((n,self.cols))
        
        for i in range(n):
            end: int = idx_set[i] - self.args.horizon + 1
            start: int = end - self.args.window
            X[i,:,:] = torch.from_numpy(self.data_np[start:end, :]) #self.dat is normalized from self.rawdat and self.rawdat is the original data given as input from the dataloader
            Y[i,:] = torch.from_numpy(self.data_np[idx_set[i], :]) #Y ends self.horizon steps ahead of X --> self.horizon is the forcasting horizon

        return [X, Y]
    
   
    def _model_loading(self, modelID: str) -> None:
        path = './models/'+ modelID+ '/model.pt'
        try: #Raise an error when the model is saved on a GPU and loaded on a CPU
            model = torch.load(path)
            self.Model = model
        except RuntimeError:
            model = torch.load(path, map_location=self.device)
            self.Model = model
            self.Model.device = self.device #Necessary for the model to know on which device it is running. It was not working with self.Model.to(self.device)

        self.Model.to(self.device)
        self.Model.BATCH_SIZE = self.batch_size
        #Copy model.pt and infotext.txt to self.savedir
        shutil.copy('./models/'+ modelID+ '/modelinfo.txt', self.savedir)
        shutil.copy('./models/'+ modelID+ '/model.pt', self.savedir)


    def _set_savedir(self, modelID: str) -> None:
        self.modelID: str = modelID
        # dir_name = self.modelID + '_AirlineID_' + str(self.args.airlineID)
        dir_name = self.modelID
        self.savedir: str = os.path.join('models/testing', dir_name)
        self.savedir = os.path.abspath(self.savedir)
        if not os.path.isdir(self.savedir):
            os.makedirs(self.savedir)   


    def _criterion_eval(self) -> None:
        if self.args.L1Loss:
            self.criterion = nn.L1Loss(size_average = False).cpu() #L1 Loss equals MAE
        else:
            self.criterion = nn.MSELoss(size_average = False).cpu()
        self.evaluateL1 = nn.L1Loss(size_average = False).cpu()
        self.evaluateL2 = nn.MSELoss(size_average = False).cpu()
        if self.args.cuda:
            self.criterion = self.criterion.cuda()
            self.evaluateL1 = self.evaluateL1.cuda()
            self.evaluateL2 = self.evaluateL2.cuda()


    def _plot_initialisation(self):
        self.train_loss_plot: list = []
        self.test_rmse_plot: list = []
        self.test_mae_plot: list = []

        self.metric_mae = [self.train_loss_plot, self.test_mae_plot]
        self.metric_rmse = [[], self.test_rmse_plot]
        self.plot_metrics = {'MAE': self.metric_mae}


    def get_batches(self, inputs: torch.Tensor, targets: torch.Tensor, batch_size: int):
        '''
        Generates a batch of samples Each sample has the shape of (window,n_features) for X and (1,n_features) for Y. The yield command indicates this function is used as a generator to
        iterate over a sequence of batches. 

        The function is used in the train.py file to generate a batch of samples for training the model.
        '''
        length = len(inputs)
        index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt] 
            Y = targets[excerpt]
            if (self.args.cuda == 'cuda'):
                X = X.cuda()
                Y = Y.cuda()
            else:
                X = X.cpu()
                Y = Y.cpu()
            yield Variable(X), Variable(Y)
            start_idx += self.batch_size


    def evaluate(self):
        X: torch.Tensor = self.eval_data[0]
        Y: torch.Tensor = self.eval_data[1]
        
        self._eval_run(self.eval_data, X, Y)   
        self._print_metrics()
        if self.args.plot: 
            self._plot_metrics() 
            
            
    def _eval_run(self, data, X: torch.Tensor, Y: torch.Tensor) -> None:
        self.Model.eval()
        total_MSE = 0
        total_MAE = 0
        total_RMSE = 0
        total_RAE = 0
        total_RSE = 0
        n_samples = 0
        n_batches = 0
        predict = torch.Tensor().to(self.device)
        test = torch.Tensor().to(self.device)
        batch_size = self.batch_size

        with torch.no_grad():
            for X, Y in self.get_batches(X, Y, batch_size=self.batch_size):
                if X.shape[0] != batch_size:
                    print(f'{n_batches} batches passed before Test Set < Batch Size --> Testing stopped')
                    break
                output = self.Model(X)

                #Selecting the sixth last column uses only the profit for the loss calculation
                scale: torch.Tensor = self.scale.expand(output.size(0), self.cols).to(self.device)
                scale = scale[0,:]
                scale_profit = scale[-1]
                # scale_profit = scale[:,-1] 

                predict = torch.cat((predict, output[0,:]*scale))
                test = torch.cat((test, Y*scale))


                output_profit = output[0,-1]
                output_unscaled = output_profit * scale_profit
                Y_profit = Y[:,-1]
                Y_profit_unscaled = Y_profit.to(self.device) * scale_profit


                total_MSE += self.evaluateL2(output_unscaled, Y_profit_unscaled).item() # L2 Loss equals MSE
                total_MAE += self.evaluateL1(output_unscaled, Y_profit_unscaled).item() # L1 Loss equals MAE
                total_RMSE += math.sqrt(self.evaluateL2(output_unscaled, Y_profit_unscaled).item()) 
                total_RAE += torch.sum(torch.abs(output_unscaled - Y_profit_unscaled) / torch.sum(torch.abs(Y_profit_unscaled - torch.mean(Y_profit_unscaled)))).item() #RAE is the relative absolute error from the test set over the naive model, data.rae,(mean of the test set)
                total_RSE += torch.sum(torch.square(output_unscaled - Y_profit_unscaled) / torch.sum(torch.square(Y_profit_unscaled - torch.mean(Y_profit_unscaled)))).item() #RSE is the relative squared error, same as RAE but squared
                n_samples += (output.size(0) * 1) #* x 1 only when predicting profit, when predicting the whole feature set then x data.cols
                n_batches += 1
                del scale, X, Y
                torch.cuda.empty_cache()
            
        predict = predict.reshape(-1,self.cols)
        self._calculate_metrics(data, total_MSE, total_MAE, total_RMSE, total_RAE, total_RSE, n_batches, predict, test)

    def _calculate_metrics(self, data: DataUtility, total_MSE, total_MAE, total_RMSE, total_RAE, total_RSE, n_batches, predict, test):
        self.metrics['RMSE'] = np.round(math.sqrt(total_RMSE / n_batches), 4)
        self.metrics['RSE'] = np.round(math.sqrt(total_RSE / n_batches), 4)
        self.metrics['RAE'] = np.round((total_RAE / n_batches), 4) 
        self.metrics['MAE'] = np.round(total_MAE / n_batches, 4)
        self.metrics['MSE'] = np.round(total_MSE / n_batches, 4)

        predict = predict.data.cpu().numpy()
        self.predict = predict
        Ytest = test.data.cpu().numpy()
        self.Ytest = Ytest
        sigma_p = (predict).std(axis=0)
        sigma_g = (Ytest).std(axis=0)
        mean_p = predict.mean(axis=0)
        mean_g = Ytest.mean(axis=0)
        index = (sigma_g != 0)
        correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
        self.metrics['Correlation'] = np.round((correlation[index]).mean(), 4)
        

    def _print_metrics(self) -> None:
        print(f'| MAE: {self.metrics["MAE"]} | MSE: {self.metrics["MSE"]} | RMSE: {self.metrics["RMSE"]} | RSE: {self.metrics["RSE"]} | RAE: {self.metrics["RAE"]} | Correlation: {self.metrics["Correlation"]}')
        


    def _plot_metrics(self) -> None:


        #Plot predict and test data

        timescale = list(self.timescale[(self.args.window+self.args.horizon-1):])
        times = ['1','2','3','4']
        timescale = times + timescale

        #! Hardcoded not for general use, only for airline with operation from 1990 to 2023
        plt.figure(figsize=(14,5))
        plt.plot(timescale[:100],self.predict[:,-1],'o-', label='Prediction')
        plt.plot(timescale[4:],self.Ytest[:,-1],'o-', label='True')

        # Change title font size
        plt.title(f'Airline {self.args.airlineID} | MAE: {self.metrics["MAE"]:.2f} | MSE: {self.metrics["MSE"]:.2f} | RMSE: {self.metrics["RMSE"]:.2f} | RSE: {self.metrics["RSE"]:.2f} | RAE: {self.metrics["RAE"]:.2f} | Correlation: {self.metrics["Correlation"]:.2f}', fontsize=10)

        plt.legend()
        plt.xticks(rotation='vertical', fontsize=8)

        # Show only every second x-label
        locs, labels = plt.xticks()  # Get locations and labels
        for i, label in enumerate(labels):
            if i < len(timescale):
                label.set_text(str(timescale[i]))
        plt.xticks(locs[::2], labels[::2])  # Set locations and labels

        plt.savefig(os.path.join(self.savedir, f'prediction_airlineID_{self.args.airlineID}.png'))

        plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('--data', type=str, default=None, help='location of the data file')
    parser.add_argument('--modelID', type=str, default=None, help='location of the model file')
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--airlineID', type=int, default=19393, help='Airline ID for which the model is evaluated. Default, Southwest')
    parser.add_argument('--normalize', type=int, default=4)
    parser.add_argument('--window', type=int, default=32, help='window size')
    parser.add_argument('--horizon', type=int, default=4, help='forecasting horizon')
    parser.add_argument('--L1Loss', type=bool, default=True)
    parser.add_argument('--plot', type=bool, default=False)

    args = parser.parse_args()

    #Airlines: American Airlines, Delta, United, Southwest, JetBlue, Alaska, Spirit, Frontier, Hawaiian, SkyWest
    airlineIDs = [19805, 19790, 19977, 19393, 20409, 19930, 20416, 20436, 19690, 20304]
    df_predictions = pd.DataFrame()
    max_time_span = range(1990,2024)
    quarters = 4
    time_line = [str(year) + '-' + str(quarter) for year in max_time_span for quarter in range(1,quarters+1)]
    df_predictions['Time'] = time_line

    if args.airlineID == 8888: #If the airlineID is 8888, the the model is evaluated for the given list of airlineIDs
        for airlineID in airlineIDs:
            args.airlineID = airlineID
            evaluation = ModelTest(args)
            evaluation.evaluate()
            time = list(evaluation.timescale[(evaluation.args.window+evaluation.args.horizon-1):])
            real_profit = evaluation.Ytest[:,-1]
            pred_profit = evaluation.predict[:,-1]
            tmp = pd.DataFrame({'Time': time, f'Profit_{airlineID}': real_profit, f'Prediction_{airlineID}': pred_profit})
            df_predictions = pd.merge(df_predictions, tmp, on='Time', how='left')
        
        filename = f'predictions_on_model_{args.modelID}_horizon{args.horizon}.csv'
        path = os.path.join(evaluation.savedir, filename)
        df_predictions.to_csv(path, index=False)

    else:
        evaluation = ModelTest(args)
        evaluation.evaluate()


