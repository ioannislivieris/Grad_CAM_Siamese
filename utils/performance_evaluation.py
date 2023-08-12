import numpy as np
from sklearn import metrics
import torch


def performance_evaluation(labels:torch.Tensor=None, predictions:torch.Tensor=None):
  '''
    Performance evaluation metrics (Accuracy, AUC, Precision, Recall, GM, CM)
    Parameters
    ----------
    labels: true tokens
    predictions: predicted tokens
    attention_mask: attention mask
    Returns
    -------
    Accuracy (float)
    AUC (float)
    Precision (float)
    Recall (float)
    GM (float)
    CM (ndarray)
  '''
  y = labels.detach().cpu().numpy().astype('int')
  pred = np.array([1 if x > 0.5 else 0 for x in predictions])

  Accuracy = 100.0*metrics.accuracy_score(y, pred)
  try:
    AUC = metrics.roc_auc_score(y, pred)
  except:
    AUC = 0.0
  # Recall = metrics.recall_score(y, pred, average='macro')
  # Precision = metrics.precision_score(y, pred, average='macro')    
  CM = metrics.confusion_matrix(y, pred)
  
  # GM = np.prod(np.diag(CM)) ** (1.0/CM.shape[0])

  return Accuracy, AUC, CM #Precision, Recall, GM