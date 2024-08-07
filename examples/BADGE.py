import numpy as np
import torch
from torch import nn
from scipy import stats
import sys
import time
from math import ceil
import cheetah_surrogate
        
def sub_select_badge(num_elements, data, candidates, path_to_model):
    """
    Args:
        num_elements: The number of elements to select and return back to the main application.
        data: A ExampleView of the data that the modes was trained with
        candidates: A ExampleView of the new data that the model has never seen
        path_to_model: Path to the latest model

    Returns:
        A tuple of input, output pairs that will be stored in the database
    """

    X, Y = candidates.get_data()

    num_items = X.shape[0]
    if num_elements > num_items:
        return X, Y

    #TODO: implement `model_loader`
    print("\n************************\nloading cheetah model\n************************\n")
    model = model_loader(path_to_model, num_inputs=X.shape[-1], num_outputs=Y.shape[-1])
    data = list(zip(X,Y))
    badge_indexes = sorted(badge_select(model, data, num_elements))
    return X[badge_indexes, ...], Y[badge_indexes, ...]

def model_loader(path, num_inputs, num_outputs):
    model = cheetah_surrogate.CheetahSurrogateDeltaUQ(num_inputs=num_inputs, num_classes=num_outputs)
    #state_dict = torch.load(path)
    #model.load_state_dict(state_dict)
    model.cuda()
    model.device = 'cuda'
    model.embedding_dim = 128
    model.target_classes = num_outputs
    return model


def badge_select(model, candidates, budget, batch_size=2048):
    """
    Selects next set of points

    Parameters
    ----------
    model: a torch model
        The pretrained model 
    candidates: An ExampleView
        The new data that the model has never seen
    budget: int
        Number of data points to select for labeling

    Returns
    ----------
    idxs: list
        List of selected data point indices with respect to unlabeled_dataset
    """ 

    model.eval()

    print(f'\nThere are {len(candidates)} points in the unlabeled dataset')
    print("\n************************\ncomputing candidate embeddings\n************************\n")
    GPU_memory_constraint = 500000
    if len(candidates)>100e6: GPU_memory_constraint = 400000
    chosen = []
    examples_seen = 0
    if len(candidates)>GPU_memory_constraint:
        print(f'\nWe detected that the unlabeled dataset may be too large to fit in memory, so we will use BADGE iteratively.')
        start = time.time()
        for chunk in range(ceil(len(candidates)/GPU_memory_constraint)):
            print(f'\nIteration {chunk}')
            candidates_chunk = candidates[chunk*GPU_memory_constraint:(chunk+1)*GPU_memory_constraint]
            gradEmbedding_chunk = get_grad_embedding(model, candidates_chunk, batch_size)
            print(f'\nThe embeddings have shape {gradEmbedding_chunk.shape}')
            print("\n************************\nselecting candidates\n************************\n")
            budget_chunk = budget * (len(gradEmbedding_chunk)/len(candidates))
            chosen_chunk = init_centers(gradEmbedding_chunk.cpu().numpy(), budget_chunk, model.device)
            chosen_chunk = list(np.array(chosen_chunk)+examples_seen)
            chosen+=chosen_chunk 
            examples_seen += len(gradEmbedding_chunk)
        assert abs(len(chosen)-budget) < chunk + 2, f'{len(chosen)} chosen, with {budget} budget, in {chunk} chunks'
        chosen = chosen[:budget]
        end = time.time()
        print(f'\nFrom {len(candidates)} points, selected {len(chosen)} in {(end-start)/60} minutes!')
        return chosen

    # We set the predict_labels argument to false to use the labels in the unlabeled_dataset object
    gradEmbedding = get_grad_embedding(model, candidates, batch_size)
    print(f'\nThe embeddings have shape {gradEmbedding.shape}')
    start = time.time()
    print("\n************************\nselecting candidates\n************************\n")
    chosen = init_centers(gradEmbedding.cpu().numpy(), budget, model.device)
    end = time.time()
    print(f'\nFrom {len(candidates)} points, selected {budget} in {(end-start)/60} minutes!')
    return chosen


def get_grad_embedding(model,
                       candidates,
                       batch_size,
                       loss_type = 'MSE'):
    
    if loss_type == 'MSE':
        loss_func = torch.nn.functional.mse_loss
    
    device = model.device
    embDim = model.embedding_dim
    target_classes = model.target_classes

    grad_embedding = torch.zeros([len(candidates), embDim * target_classes]).to(device)

    # Create a dataloader object to load the dataset
    dataloader = torch.utils.data.DataLoader(candidates, 
                                            batch_size = batch_size,
                                            shuffle = False)  

    evaluated_instances = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        start_slice = evaluated_instances
        end_slice = start_slice + inputs.shape[0]
        inputs, targets = inputs.to(device, non_blocking=True).float(), targets.to(device, non_blocking=True).float()
        print(f'input shape is {inputs.shape}, target shape is {targets.shape}')
        out, l1 = model(inputs, last=True, freeze=True)

        # Calculate loss as a sum, allowing for the calculation of the gradients using autograd wprt the outputs (bias gradients)
        loss = loss_func(out, targets, reduction="sum")
        l0_grads = torch.autograd.grad(loss, out)[0]

        l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
        l1_grads = l0_expand * l1.repeat(1, target_classes)
        grad_embedding[start_slice:end_slice] = l1_grads

        evaluated_instances = end_slice

        # Empty the cache as the gradient embeddings could be very large
        torch.cuda.empty_cache()

    # Return final gradient embedding
    return grad_embedding

def init_centers(X, K, device):
    pdist = nn.PairwiseDistance(p=2)
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pdist(torch.from_numpy(X).to(device), torch.from_numpy(mu[-1]).to(device))
            D2 = torch.flatten(D2)
            D2 = D2.cpu().numpy().astype(float)
        else:
            newD = pdist(torch.from_numpy(X).to(device), torch.from_numpy(mu[-1]).to(device))
            newD = torch.flatten(newD)
            newD = newD.cpu().numpy().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
                    
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll
