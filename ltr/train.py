import torch
import numpy as np

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from ltr.logging_policy import LoggingPolicy
from ltr.loss import pointwise_loss, pairwise_loss, listwise_loss, compute_lambda_i, listNet_loss, unbiased_listNet_loss
from ltr.dataset import LTRData, QueryGroupedLTRData, qg_collate_fn, ClickLTRData
from ltr.eval import evaluate_model


def train_batch(net, x, y, loss_fn, optimizer):
    """
    Performs a single training batch update for the given neural network.

    Parameters
    ----------
    net : torch.nn.Module
        The neural network model to be trained.
    x : torch.Tensor
        Input tensor containing the training data.
    y : torch.Tensor
        Target tensor containing the ground truth labels.
    loss_fn : callable
        Loss function used to compute the training loss.
    optimizer : torch.optim.Optimizer
        Optimizer used for updating the model parameters.

    Returns
    -------
    None
    """
    optimizer.zero_grad()
    out = net(x)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()


# TODO: Implement this!
def train_pointwise(net, params, data):
    """
    This function should train a Pointwise network.

    The network is trained using the Adam optimizer


    Note: Do not change the function definition!


    Hints:
    1. Use the LTRData class defined above
    2. Do not forget to use net.train() and net.eval()

    Inputs:
            net: the neural network to be trained

            params: params is an object which contains config used in training
                (eg. params.epochs - the number of epochs to train).
                For a full list of these params, see the next cell.

    Returns: a dictionary containing: "metrics_val" (a list of dictionaries) and
             "metrics_train" (a list of dictionaries).

             "metrics_val" should contain metrics (the metrics in params.metrics) computed
             after each epoch on the validation set (metrics_train is similar).
             You can use this to debug your models.

    """

    val_metrics_epoch = []
    train_metrics_epoch = []
    optimizer = Adam(net.parameters(), lr=params.lr)
    loss_fn = pointwise_loss

    ### BEGIN SOLUTION
    # Step Tips:
    # Step 1: Create a DataLoader for the training data
    # Step 2: Iterate over the data for the number of epochs
    # Step 3: Iterate over each batch of data and use the train_batch function to train the model
    # Step 4: At the end of the epoch, evaluate the model on the data using the evaluate_model function (bot train and val)
    # Step 5: Append the metrics to train_metrics_epoch and val_metrics_epoch
    # Create DataLoader for training data
    train_dl = DataLoader(LTRData(data, "train"), batch_size=params.batch_size, shuffle=True)
    
    # Loop through training epochs
    for epoch in range(params.epochs):
        # Set model to training mode
        net.train()
        
        # Process each batch of training data
        for x, y in train_dl:
            train_batch(net, x, y, loss_fn, optimizer)
        
        # After each epoch, evaluate model performance
        net.eval()  # Switch to evaluation mode
        
        # Collect metrics on training data
        train_metrics = evaluate_model(data, net, "train")
        train_metrics_epoch.append(train_metrics)
        
        # Collect metrics on validation data
        val_metrics = evaluate_model(data, net, "validation")
        val_metrics_epoch.append(val_metrics)
    ### END SOLUTION

    return {"metrics_val": val_metrics_epoch, "metrics_train": train_metrics_epoch}


# TODO: Implement this!
def train_batch_vector(net, x, y, loss_fn, optimizer):
    """
    Takes as input a batch of size N, i.e. feature matrix of size (N, NUM_FEATURES), label vector of size (N), the loss function and optimizer for computing the gradients, and updates the weights of the model.
    The loss function returns a vector of size [N, 1], the same as the output of network.

    Input:  x: feature matrix, a [N, NUM_FEATURES] tensor
            y: label vector, a [N] tensor
            loss_fn: an implementation of a loss function
            optimizer: an optimizer for computing the gradients (we use Adam)
    """
    ### BEGIN SOLUTION
    # Step tips:
    # Step 1: Zero the gradients of the optimizer
    # Step 2: Forward pass the input through the network using the net
    # Step 3: Compute the loss using the loss_fn
    # Step 4: Backward pass to compute the gradients
    # Step 5: Update the weights using the optimizer

    

    ### END SOLUTION


# TODO: Implement this!
def train_pairwise(net, params, data):
    """
    This function should train the given network using the pairwise loss

    Returns: a dictionary containing: "metrics_val" (a list of dictionaries) and
             "metrics_train" (a list of dictionaries).

             "metrics_val" should contain metrics (the metrics in params.metrics) computed
             after each epoch on the validation set (metrics_train is similar).
             You can use this to debug your models

    Note: Do not change the function definition!
    Note: You can assume params.batch_size will always be equal to 1

    Hint: Consider the case when the loss function returns 'None'

    net: the neural network to be trained

    params: params is an object which contains config used in training
        (eg. params.epochs - the number of epochs to train).
        For a full list of these params, see the next cell.
    """

    val_metrics_epoch = []
    train_metrics_epoch = []

    ### BEGIN SOLUTION
    # Step Tips:
    # Step 1: Create a DataLoader for the training data
    # Step 2: Create your Adam optimizer
    # Step 3: Iterate over the data for the number of epochs
    # Step 4: Iterate over each batch of data and compute the scores using the forward pass of the network
    # Step 5: Compute the pairwise loss using the pairwise_loss function
    # Step 6: Compute the gradients and update the weights using the optimizer
    # Step 7: At the end of the epoch, evaluate the model on the data using the evaluate_model function (both train and val)
    # Step 8: Append the metrics to train_metrics_epoch and val_metrics_epoch

    # Step 1: Create a DataLoader for the training data
    train_dl = DataLoader(
        QueryGroupedLTRData(data, "train"),
        batch_size=params.batch_size,  # Will be 1
        shuffle=True,
        collate_fn=qg_collate_fn
    )

    # Step 2: Create your Adam optimizer
    optimizer = Adam(net.parameters(), lr=params.lr)

    # Step 3: Iterate over the data for the number of epochs
    for epoch in range(params.epochs):
        # Set model to training mode
        net.train()
        
        # Step 4: Iterate over each batch of data and compute the scores using the forward pass of the network
        for features_list, labels_list in train_dl:
            features = features_list[0]  # Documents for this query
            labels = labels_list[0]      # Relevance labels for these documents
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass to get predicted scores
            scores = net(features)
            
            # Step 5: Compute the pairwise loss using the pairwise_loss function
            loss = pairwise_loss(scores, labels)
            
            # Skip this query if loss is None (happens if query has only one document)
            if loss is None:
                continue
                
            # Step 6: Compute the gradients and update the weights using the optimizer
            loss.backward()
            optimizer.step()
        
        # Step 7: At the end of the epoch, evaluate the model on the data using the evaluate_model function (both train and val)
        net.eval()
        
        # Get metrics on training data
        train_metrics = evaluate_model(data, net, "train")
        train_metrics_epoch.append(train_metrics)
        
        # Get metrics on validation data
        val_metrics = evaluate_model(data, net, "validation")
        val_metrics_epoch.append(val_metrics)
    ### END SOLUTION

    return {"metrics_val": val_metrics_epoch, "metrics_train": train_metrics_epoch}


# TODO: Implement this!
def train_pairwise_spedup(net, params, data):
    """
    This function should train the given network using the sped up pairwise loss


    Note: Do not change the function definition!
    Note: You can assume params.batch_size will always be equal to 1


    net: the neural network to be trained

    params: params is an object which contains config used in training
        (eg. params.epochs - the number of epochs to train).
        For a full list of these params, see the next cell.

    Returns: a dictionary containing: "metrics_val" (a list of dictionaries) and
             "metrics_train" (a list of dictionaries).

             "metrics_val" should contain metrics (the metrics in params.metrics) computed
             after each epoch on the validation set (metrics_train is similar).
             You can use this to debug your models
    """

    val_metrics_epoch = []
    train_metrics_epoch = []

    ### BEGIN SOLUTION
    # Step Tips:
    # Step 1: Create a DataLoader for the training data
    # Step 2: Create your Adam optimizer
    # Step 3: Iterate over the data for the number of epochs
    # Step 4: Iterate over each batch of data and compute the scores using the forward pass of the network
    # Step 5: Compute the lambda gradient values for the pairwise loss (spedup) with the compute_lambda_i method on the scores and the output labels
    # Step 6: Bacward from the scores with the use of the lambda gradient values
    # Step 7: Update the weights using the optimizer
    # Step 8: At the end of the epoch, evaluate the model on the data using the evaluate_model function (both train and val)
    # Step 9: Append the metrics to train_metrics_epoch and val_metrics_epoch
    # Step 1: Create a DataLoader for the training data
    train_dl = DataLoader(
        QueryGroupedLTRData(data, "train"),
        batch_size=params.batch_size,  # Will be 1
        shuffle=True,
        collate_fn=qg_collate_fn
    )

    # Step 2: Create your Adam optimizer
    optimizer = Adam(net.parameters(), lr=params.lr)

    # Step 3: Iterate over the data for the number of epochs
    for epoch in range(params.epochs):
        # Set model to training mode
        net.train()
        
        # Step 4: Iterate over each batch of data and compute the scores
        for features_list, labels_list in train_dl:
            features = features_list[0]  # Extract features for the query
            labels = labels_list[0]      # Extract labels for the query
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass to get scores
            scores = net(features)
            
            # Step 5: Compute lambda gradients directly
            lambda_i = compute_lambda_i(scores, labels)
            
            # Step 6: Backward from scores using lambda gradients
            scores.backward(lambda_i)
            
            # Step 7: Update the weights
            optimizer.step()
        
        # Step 8: Evaluate model after each epoch
        net.eval()
        
        # Get metrics on training data
        train_metrics = evaluate_model(data, net, "train")
        train_metrics_epoch.append(train_metrics)
        
        # Get metrics on validation data
        val_metrics = evaluate_model(data, net, "validation")
        val_metrics_epoch.append(val_metrics)
    
    ### END SOLUTION

    return {"metrics_val": val_metrics_epoch, "metrics_train": train_metrics_epoch}


# TODO: Implement this!
def train_listwise(net, params, data):
    """
    This function should train the given network using the listwise (LambdaRank) loss

    Note: Do not change the function definition!
    Note: You can assume params.batch_size will always be equal to 1


    net: the neural network to be trained

    params: params is an object which contains config used in training
        (eg. params.epochs - the number of epochs to train).
        For a full list of these params, see the next cell.

    Returns: a dictionary containing: "metrics_val" (a list of dictionaries) and
             "metrics_train" (a list of dictionaries).

             "metrics_val" should contain metrics (the metrics in params.metrics) computed
             after each epoch on the validation set (metrics_train is similar).
             You can use this to debug your models
    """

    val_metrics_epoch = []
    train_metrics_epoch = []

    ### BEGIN SOLUTION
    # Step Tips:
    # Step 1: Create a DataLoader for the training data
    # Step 2: Create your Adam optimizer
    # Step 3: Iterate over the data for the number of epochs
    # Step 4: Iterate over each batch of data and compute the scores using the forward pass of the network
    # Step 5: Compute the lambda gradient values for the listwise (LambdaRank) loss with the compute_lambda_i method on the scores and the output labels
    # Step 6: Bacward from the scores with the use of the lambda gradient values
    # Step 7: Update the weights using the optimizer
    # Step 8: At the end of the epoch, evaluate the model on the data using the evaluate_model function (both train and val)
    # Step 9: Append the metrics to train_metrics_epoch and val_metrics_epoch

    # Step 1: Create a DataLoader for the training data
    train_dl = DataLoader(
        QueryGroupedLTRData(data, "train"),
        batch_size=params.batch_size,  # Will be 1
        shuffle=True,
        collate_fn=qg_collate_fn
    )

    # Step 2: Create your Adam optimizer
    optimizer = Adam(net.parameters(), lr=params.lr)

    # Step 3: Iterate over the data for the number of epochs
    for epoch in range(params.epochs):
        # Set model to training mode
        net.train()
        
        # Step 4: Iterate over each batch of data and compute the scores
        for features_list, labels_list in train_dl:
            features = features_list[0]  # Documents for this query
            labels = labels_list[0]      # Relevance labels for these documents
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass to get scores
            scores = net(features)
            
            # Step 5: Compute lambda gradient values for listwise loss
            lambda_i = listwise_loss(scores, labels)
            
            # Step 6: Backward using lambda gradients
            scores.backward(lambda_i)
            
            # Step 7: Update weights
            optimizer.step()
        
        # Step 8: Evaluate model performance after each epoch
        net.eval()
        
        # Get metrics on training data
        train_metrics = evaluate_model(data, net, "train")
        train_metrics_epoch.append(train_metrics)
        
        # Get metrics on validation data
        val_metrics = evaluate_model(data, net, "validation")
        val_metrics_epoch.append(val_metrics)
    ### END SOLUTION

    return {"metrics_val": val_metrics_epoch, "metrics_train": train_metrics_epoch}


# TODO: Implement this!
def logit_to_prob(logit):
    """
    Converts logits to probabilities using the sigmoid function.

    Parameters
    ----------
    logit : torch.Tensor
        Input tensor containing logit values.

    Returns
    -------
    torch.Tensor
        Output tensor containing probabilities in the range [0,1].
    """

    ### BEGIN SOLUTION
    return torch.sigmoid(logit)
    ### END SOLUTION


# TODO: Implement this!
def train_biased_listNet(net, params, data):
    """
    This function should train the given network using the (biased) listNet loss

    Note: Do not change the function definition!
    Note: You can assume params.batch_size will always be equal to 1


    net: the neural network to be trained

    params: params is an object which contains config used in training
            params.epochs - the number of epochs to train.
            params.lr - learning rate for Adam optimizer.
            params.batch_size - batch size (always equal to 1)

    Returns: a dictionary containing: "metrics_val" (a list of dictionaries).

             "metrics_val" should contain metrics (the metrics in params.metrics) computed
             after each epoch on the validation set.
             You can use this to debug your models
    """

    val_metrics_epoch = []

    assert params.batch_size == 1

    # logging_policy = LoggingPolicy()
    if hasattr(params, "policy_path"):
        logging_policy = LoggingPolicy(params.policy_path)
    else:
        logging_policy = LoggingPolicy()

    ### BEGIN SOLUTION
    # Step 1: Create the train data loader
    # Step 2: Create the validation data loader
    # Step 3: Create the Adam optimizer
    # Step 4: Iterate over the epochs and data entries
    # Step 5: Train the model using the listNet loss
    # Step 6: Evaluate on the validation set every epoch
    # Step 7: Store the metrics in val_metrics_epoch


    # Step 1: Create the train data loader
    train_dl = DataLoader(
        ClickLTRData(data, logging_policy),
        batch_size=params.batch_size,  # Will be 1
        shuffle=True
    )

    # Step 2: Create the Adam optimizer
    optimizer = Adam(net.parameters(), lr=params.lr)

    # Step 3: Iterate over the epochs and data entries
    for epoch in range(params.epochs):
        # Set model to training mode
        net.train()
        
        # Step 4: Process each batch of training data
        for features, clicks, positions in train_dl:
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = net(features)
            
            # Step 5: Calculate loss using listNet loss
            loss = listNet_loss(output, clicks)
            
            # Backward pass and update weights
            loss.backward()
            optimizer.step()
        
        # Step 6: Evaluate on the validation set after each epoch
        net.eval()
        
        # Get metrics on validation data
        val_metrics = evaluate_model(data, net, "validation")
        val_metrics_epoch.append(val_metrics)
        
        
    ### END SOLUTION

    return {"metrics_val": val_metrics_epoch}


# TODO: Implement this!
def train_unbiased_listNet(net, params, data):
    """
    This function should train the given network using the unbiased_listNet loss

    Note: Do not change the function definition!
    Note: You can assume params.batch_size will always be equal to 1
    Note: For this function, params should also have the propensity attribute


    net: the neural network to be trained

    params: params is an object which contains config used in training
            params.epochs - the number of epochs to train.
            params.lr - learning rate for Adam optimizer.
            params.batch_size - batch size (always equal to 1)
            params.propensity - the propensity values used for IPS in unbiased_listNet

    Returns: a dictionary containing: "metrics_val" (a list of dictionaries).

             "metrics_val" should contain metrics (the metrics in params.metrics) computed
             after each epoch on the validation set.
             You can use this to debug your models
    """

    val_metrics_epoch = []

    assert params.batch_size == 1
    assert hasattr(params, "propensity")

    # logging_policy = LoggingPolicy()
    if hasattr(params, "policy_path"):
        logging_policy = LoggingPolicy(params.policy_path)
    else:
        logging_policy = LoggingPolicy()

    ### BEGIN SOLUTION
    # Step 1: Create the train data loader
    # Step 2: Create the validation data loader
    # Step 3: Create the Adam optimizer
    # Step 4: Iterate over the epochs and data entries
    # Step 5: Train the model using the unbiased listNet loss
    # Step 6: Evaluate on the validation set every epoch
    # Step 7: Store the metrics in val_metrics_epoch

    # Step 1: Create the train data loader
    train_dl = DataLoader(
        ClickLTRData(data, logging_policy),
        batch_size=params.batch_size,  # Will be 1
        shuffle=True
    )

    # Step 2: Create the Adam optimizer
    optimizer = Adam(net.parameters(), lr=params.lr)

    # Step 3: Iterate over the epochs and data entries
    for epoch in range(params.epochs):
        # Set model to training mode
        net.train()
        
        # Step 4: Process each batch of training data
        for features, clicks, positions in train_dl:
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = net(features)
            
            # Step 5: Calculate unbiased loss using propensity scores
            # Get propensity scores for current positions
            propensity_scores = params.propensity[positions]
            
            # Use unbiased version of ListNet loss that incorporates propensity scores
            loss = unbiased_listNet_loss(output, clicks, propensity_scores)
            
            # Backward pass and update weights
            loss.backward()
            optimizer.step()
        
        # Step 6: Evaluate on the validation set after each epoch
        net.eval()
        
        # Get metrics on validation data
        val_metrics = evaluate_model(data, net, "validation")
        val_metrics_epoch.append(val_metrics)
        
        
    ### END SOLUTION

    return {"metrics_val": val_metrics_epoch}