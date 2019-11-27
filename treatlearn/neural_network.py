"""
Neural Network Architectures for Treatment Effect Estimation
"""

import torch

from torch.utils.data import Dataset

import torch.nn as nn
from torch.autograd import Variable


# TODO: I think a batch sampler stratified on g makes more sense.

class TabularData(Dataset):
    def __init__(self, X, y):
        """
        Torch data Loader for experimental data

        X : array-like, shape (n_samples, n_features)
          The input data.

        y : array-like, shape (n_samples,)
          The target values (class labels in classification, real numbers in regression).

        g : array-like
          The group indicator (e.g. 0 for control group, 1 for treatment group)
        """
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]

class ExperimentData(Dataset):
    def __init__(self, X, y, g):
        """
        Torch data Loader for experimental data

        X : array-like, shape (n_samples, n_features)
          The input data.

        y : array-like, shape (n_samples,)
          The target values (class labels in classification, real numbers in regression).

        g : array-like
          The group indicator (e.g. 0 for control group, 1 for treatment group)
        """
        self.X = X
        self.y = y
        self.g = g

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx], self.g[idx]




# class ExperimentPurchaseData(Dataset):
#     def __init__(self, X, y, c, g):
#         self.X = X
#         self.c = c
#         self.y = y
#         self.g = g
        
#     def __len__(self):
#         return self.X.shape[0]
    
#     def __getitem__(self, idx):
#         return self.X[idx,:], self.y[idx], self.c[idx], self.g[idx]
                







class NNet(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes, output_dim, loss, output_activation=None):
        """
        input_dim : int
          Number of input variables

        hidden_layer_sizes : list of int
          List of the number of nodes in each fully connected hidden layer.
          Can be an empty list to specify no hidden layer

        loss : pyTorch loss

        sigmoid : boolean
          Sigmoid activation on the output node?
        """
        super().__init__()

        self.input_dim = input_dim
        self.layer_sizes = hidden_layer_sizes
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.iter = 0
        # The loss function could be MSE or BCELoss depending on the problem
        self.lossFct = loss

        # We leave the optimizer empty for now to assign flexibly
        self.optim = None

        hidden_layer_sizes = [input_dim] + hidden_layer_sizes
        last_layer = nn.Linear(hidden_layer_sizes[-1], output_dim)
        self.layers =\
            [nn.Sequential(nn.Linear(input_, output_), nn.ReLU())
             for input_, output_ in
             zip(hidden_layer_sizes, hidden_layer_sizes[1:])] +\
            [last_layer]
        
        # The output activation depends on the problem
        if output_activation == "sigmoid":
            self.layers = self.layers + [nn.Sigmoid()]

        if output_activation == "exponential":
            class exponential(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                def forward(self, x):
                    return torch.exp(x)
            
            self.layers = self.layers + [exponential()]

        self.layers = nn.Sequential(*self.layers)

        
    def forward(self, x):
        x = self.layers(x)
        return x
    
    def fit(self, data_loader, epochs, validation_data=None):

        for epoch in range(epochs):
            running_loss = self._train_iteration(data_loader)
            val_loss = None
            if validation_data is not None:
                y_hat = self(validation_data['X'])
                val_loss = self.lossFct(input=y_hat, target=validation_data['y']).detach().numpy()
                print('[%d] loss: %.3f | validation loss: %.3f' %
                  (epoch + 1, running_loss, val_loss))
            else:
                print('[%d] loss: %.3f' %
                  (epoch + 1, running_loss))
            
            
                
    def _train_iteration(self,data_loader):
        running_loss = 0.0
        for i, (X,y,g) in enumerate(data_loader):
            
            X = X.float()
            y = y.unsqueeze(1).float()
            
            X = Variable(X, requires_grad=True)
            y = Variable(y)
                      
            pred = self(X)
            loss = self.lossFct(pred, y)
            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            running_loss += loss.item()
               
        return running_loss
    
    def predict(self, X):
        X = torch.Tensor(X)
        return self(X).detach().numpy().squeeze()





class HurdleNet(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes, output_dim, loss, loss_p=None, p_weight=1,
                joint_layer_sizes=None):
        """
        input_dim : int
          Number of input variables

        hidden_layer_sizes : list of int
          List of the number of nodes in each fully connected hidden layer.
          Can be an empty list to specify no hidden layer

        loss : pyTorch loss
          Loss function for joint training of the Hurdle network

        loss_p : pyTorch loss
          Optional loss function for the hurdle module to add to the overall
          loss function. Ensures that hurdle probabilities are calibrated.

        p_weight : int
          Weight on loss_p before adding to overall loss. Default is loss + 1 * loss_p

        """
        super().__init__()

        self.input_dim = input_dim
        self.layer_sizes = hidden_layer_sizes
        self.joint_layer_sizes = joint_layer_sizes
        self.output_dim = output_dim
        self.iter = 0
        self.optim = None
        self.lossFct_p = loss_p
        self.lossFct = loss
        self.p_weight = p_weight
        
        if joint_layer_sizes:
          self.joint_layers =\
              nn.Sequential(*[nn.Sequential(nn.Linear(input_, output_), nn.ReLU())
              for input_, output_ in
              zip(joint_layer_sizes, joint_layer_sizes[1:])])
          input_dim = joint_layer_sizes[-1]

        self.p_net = NNet(input_dim, hidden_layer_sizes, output_dim, output_activation="sigmoid", loss=loss_p)
        self.v_net = NNet(input_dim, hidden_layer_sizes, output_dim, loss=loss) #, output_activation="exponential"


    def forward(self, x):
        if self.joint_layer_sizes:
            x = self.joint_layers(x)
        #for layer in self.layers[:-1]:
        #    x = F.relu(layer(x))
        #x = self.layers[-1](x)
        p = self.p_net(x)
        v = self.v_net(x)
        #v = nn.functional.softplus(v)

        y = p*v

        return y, p, v


    def fit(self, data_loader, epochs, validation_data=None):

        for epoch in range(epochs):
            running_loss = self._train_iteration(data_loader)
            val_loss = None
            if validation_data is not None:
                y_hat, c_hat, _ = self(validation_data['X'])
                val_loss = self.lossFct(input=y_hat, target=validation_data['y']).detach().numpy()
                if self.lossFct_p:
                    val_loss_p = self.lossFct_p(input=c_hat, target=validation_data['c']).detach().numpy()
                    print('[%d] loss: %.3f | validation loss: %.3f | val loss class: %.3f' %
                    (epoch + 1, running_loss, val_loss, val_loss_p))
                else:
                    print('[%d] loss: %.3f | validation loss: %.3f' %
                    (epoch + 1, running_loss, val_loss))
            else:
                print('[%d] loss: %.3f' %
                  (epoch + 1, running_loss))
            
            
    def _train_iteration(self,data_loader):
        running_loss = 0.0
        for i, (X, y, c) in enumerate(data_loader):

            X = X.float()
            c = c.float()
            y = y.float()
            #c = c.unsqueeze(1).float()
            #y = y.unsqueeze(1).float()

            X = Variable(X)
            c = Variable(c)
            y = Variable(y)
     
            y_hat, c_hat, _ = self(X)

            loss = self.lossFct(input=y_hat.squeeze(), target=y)

            if self.lossFct_p:
                loss_p = self.lossFct_p(input=c_hat.squeeze(), target=c)
                loss = loss + self.p_weight * loss_p 

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            running_loss += loss.item()

        return running_loss

    def predict(self, X):
        """
        Predict the (continuous) outcome for observations X
        """
        X = torch.Tensor(X).float()
        return self(X)[0].detach().numpy().squeeze()

    def predict_proba(self, X):
        """
        Predict the binary first-stage outcome for observations X
        """
        x = torch.Tensor(X)

        if self.joint_layer_sizes:
            x = self.joint_layers(x)
        prob = self.p_net(x)
        return torch.cat((1-prob,prob), dim=1).detach().numpy()