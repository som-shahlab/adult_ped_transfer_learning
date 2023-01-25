import pandas as pd
import numpy as np

def bootstrap_noninferiority_test(
    treatment:np.ndarray, 
    control:np.ndarray, 
    margin:float=0.05, 
    n_boot:int=10000,
    seed:int=444,
    higher_is_better:bool=True,
    )->(float,float,list):
    """
    
    """
    assert(len(treatment)==len(control))
    
    np.random.seed(seed)
    diffs = np.empty(n_boot)
    diffs[:] = np.nan
    
    for i in range(n_boot):
        ids = np.random.choice(len(treatment),len(treatment),replace=True)
        diffs[i] = np.mean(treatment[ids] - control[ids])
    
    if higher_is_better:
        p = np.sum(diffs<(-margin*np.mean(control)))/len(diffs) + 0.5*np.sum(diffs==(-margin*np.mean(control)))/len(diffs)
    else:
        p = np.sum(diffs>(-margin*np.mean(control)))/len(diffs) + 0.5*np.sum(diffs==(-margin*np.mean(control)))/len(diffs)
        
    return (2*min(p, 1-p), np.mean(diffs), np.percentile(diffs,[2.5,97.5]), diffs)

def bootstrap_superiority_test(
    treatment:np.ndarray, 
    control:np.ndarray, 
    n_boot:int=10000,
    seed:int=444,
    higher_is_better:bool=True,
    )->(float,float,list):
    """
    
    """
    assert(len(treatment)==len(control))
    
    np.random.seed(seed)
    diffs = np.empty(n_boot)
    diffs[:] = np.nan
    
    for i in range(n_boot):
        ids = np.random.choice(len(treatment),len(treatment),replace=True)
        diffs[i] = np.mean(treatment[ids] - control[ids])
    
    if higher_is_better:
        p = np.sum(diffs<=0)/len(diffs)
    else:
        p = np.sum(diffs>=0)/len(diffs)
        
    return (2*min(p, 1-p), np.mean(diffs), np.percentile(diffs,[2.5,97.5]), diffs)