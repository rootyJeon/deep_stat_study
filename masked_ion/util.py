import torch
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt

def make_train_step(model, loss_fn, optimizer):
    def train_step_fn(x, y):

        model.train()
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()
    return train_step_fn

def make_valid_step(model, loss_fn, verbose=False):
    
    def valid_step_fn(x, y):
        model.eval()
        y_hat = model(x)        
        model_prediction = torch.argmax(y_hat[0], axis=1)
        target = torch.argmax(y[0], axis=1)
        correct_cnt = torch.sum(model_prediction == target)
        loss_classifier, loss_regressor = loss_fn(y_hat, y)
        
        if verbose:
            return loss_classifier.item(), loss_regressor.item(), target, model_prediction
        else:
            return loss_classifier.item(), loss_regressor.item(), correct_cnt.item()
    return valid_step_fn

def draw_confusion_matrix(label_list, pred_list):
    num_classes = 4
    classes = ['5', '6', '7', '8']

    label_list = np.concatenate(label_list)
    pred_list = np.concatenate(pred_list)

    confusion_matrix = metrics.confusion_matrix(label_list, pred_list, labels=[i for i in range(num_classes)])
    confusion_matrix = np.round(confusion_matrix / len(label_list), 2)

    total_correct = np.sum(pred_list == label_list)
    accuracy = np.round(total_correct / len(pred_list) * 100, 3)
    
    # Create confusion matrix plot
    img = sns.heatmap(confusion_matrix, 
                      annot=True, 
                      cmap='Blues', 
                      xticklabels=classes, 
                      yticklabels=classes)
    
    # Set plot labels
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix - Accuracy: {accuracy}')
    fig = img.get_figure()
    fig.savefig('test_experiment.png')

    return accuracy

# early stopping to prevent overfitting

# simple Early Stopper
class EarlyStopper:
    def __init__(self, patience=3, delta=0.01): 
        self.patience = patience  # 이 횟수 보다 성능이 개선되지 않는 경우 학습 종료
        self.val_loss_min = np.Inf
        self.delta = delta  # 성능 개선으로 인정되는 최소 값
        self.counter = 0
        
    def __call__(self, valid_loss):
        if valid_loss < self.val_loss_min - self.delta:  # 성능이 개선되었는지 확인
            self.counter = 0
            self.val_loss_min = valid_loss
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            
        return False
    
# WeightedRandomSampler에서 사용할 label 별 weights 계산
def make_weights(class_vector):
    classes, class_counts = np.unique(class_vector, return_counts=True)  # class 및 class 별 개수 계산
    class_weights = 1. / class_counts  # class 별 weight 계산
    weights_dict = dict(zip(classes, class_weights))
    weights = [weights_dict[class_vector[i]] for i in range(len(class_vector))]
    
    return weights