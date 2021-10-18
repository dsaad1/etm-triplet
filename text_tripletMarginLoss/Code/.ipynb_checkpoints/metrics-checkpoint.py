import numpy as np
from sklearn.metrics import roc_auc_score

def getAUROC_all(outputs, targets):
    predictions = outputs.cpu().numpy()
    targets = targets.cpu().numpy()

    return roc_auc_score(targets, predictions)

def getAccuracy_all(predictions, targets):
  predictions = predictions.cpu().numpy()
  targets = targets.cpu().numpy()
  new_predictions = []
  correct = 0

  bins = [0, 1/2, 1]
  bin_indices = np.digitize(predictions, bins)
  for index in bin_indices:
    if index == 1:
      new_predictions.append(0)
    elif index == 2:
      new_predictions.append(1) 
  
  results = np.equal(new_predictions,targets)
  for result in results:
    if result:
      correct += 1
  
  return (correct / predictions.size)
