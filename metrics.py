import torchmetrics
import torchmetrics.classification
import torch
from ipdb import set_trace as bp

def subtracted_list(list1, list2):
    subtracted = list()
    for item1, item2 in zip(list1, list2):
        item = item1 - item2
        subtracted.append(item)
    return subtracted

class AvgDepScore(torchmetrics.Metric): 
    def __init__(self, dist_sync_on_step=False): 
        super().__init__()
        self.add_state("zung_scores", default=[], dist_reduce_fx="cat")

    def update(self, raw_predictions, raw_labels: torch.tensor):
        self.pred_classes = torch.argmax(raw_predictions, dim=1)
        self.zung_scores.extend(self.pred_classes * raw_labels) # raw labels of those predicted positive

    def compute(self):
        #'positive_scores' name is misleading: again, it's the raw zung scores of the patients for whom the model predicted positive, regardless of whether they really are or not
        positive_scores = [zung_score.item() for zung_score in self.zung_scores if zung_score != 0]
        if len(positive_scores) == 0:
            return 0
        else:
            return sum(positive_scores) / len(positive_scores)

    def reset(self):
        self.zung_scores = [] # reset to empty list

class CustomMAE(torchmetrics.Metric):
    def __init__(self, threshold, dist_sync_on_step=False):
        super().__init__()
        self.threshold = threshold
        self.add_state("preds_above_threshold", default=[], dist_reduce_fx="cat") # note: predictions of the guys with actual labels >=5, regardless of prediction itself
        self.add_state("labels_above_threshold", default=[], dist_reduce_fx="cat")
    
    def update(self, raw_predictions, raw_labels: torch.tensor):
        idx_above_threshold = [idx for idx, label in enumerate(raw_labels) if label >= self.threshold]
        self.preds_above_threshold.extend([raw_predictions[idx] for idx in idx_above_threshold])
        self.labels_above_threshold.extend([raw_labels[idx] for idx in idx_above_threshold])
        #bp()
    
    def compute(self):
        raw_errors = subtracted_list(self.preds_above_threshold, self.labels_above_threshold)
        abs_errors = [abs(error) for error in raw_errors]
        return sum(abs_errors) / len(abs_errors)

    def reset(self):
        self.preds_above_threshold = []
        self.labels_above_threshold = []

class Metrics():
    def __init__(self, args):
        self.args = args 
        
        # if self.args.label == 'antidep':
        #      self.num_classes = 4
        # elif self.args.task == 'multiclass':
        #     self.num_classes = 2
        # elif self.args.target == 'hy':
        #     self.num_classes = 8
        # elif self.args.target == 'age':
        #     self.num_classes = 2
        self.num_classes = args.num_classes
        if self.args.label == "dep":
            self.num_classes = 2
            if self.args.dataset == "wsc":
                self.threshold = 36
            elif self.args.dataset == "shhs2":
                self.threshold = 5
            

            
        self.init_metrics()
    
    def multiclass_prediction(self, predictions, labels, threshold=0):
        predictions = torch.argmax(predictions,dim=1)
        if self.args.label=="dep":
            labels = (labels >= threshold).int()
        return predictions, labels
    def regression_prediction(self, predictions, labels):
        threshold = 36.0 # or change to 50...but otherwise v few positive labels
        predictions = torch.where(predictions < threshold, torch.tensor(0), torch.tensor(1))
        labels = torch.where(labels < threshold, torch.tensor(0), torch.tensor(1))
        return predictions, labels
    def age_prediction(self, predictions, labels, threshold=5):
        output = torch.abs(predictions - labels) < threshold
        return output.int(), torch.zeros_like(output).to(self.args.device) + 1
    def nearest_hy_score(self, predictions, labels):
        hy_scores = torch.tensor([0, 1, 1.5, 2, 2.5, 3, 4, 5])
        output = []
        for score in predictions:
            output.append(torch.argmin(torch.abs(hy_scores - score.item())).item())
        return torch.tensor(output).to(self.args.device), labels
        
    def init_metrics(self):
        
        if self.args.task == 'multiclass':
            classifier_metrics_dict = {
                "acc": torchmetrics.Accuracy(task='multiclass',num_classes=self.num_classes).to(self.args.device),

                "kappa": torchmetrics.CohenKappa(task='multiclass',num_classes=self.num_classes).to(self.args.device),

                #"prec": torchmetrics.Precision(task = "multiclass",num_classes=self.num_classes, average = None).to(self.args.device),

                #"recall": torchmetrics.Recall(task = "multiclass",num_classes=self.num_classes, average = None).to(self.args.device),

                "f1_macro": torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average = "macro").to(self.args.device),

                "f1_c": torchmetrics.classification.MulticlassF1Score(num_classes=self.num_classes, average = None).to(self.args.device),

                "auroc": torchmetrics.AUROC(task="multiclass", num_classes=self.num_classes).to(self.args.device)

            } 
            if self.args.label=="dep":
                classifier_metrics_dict["avg_dep_score"] = AvgDepScore().to(self.args.device)
        elif self.args.task == 'binary':
            classifier_metrics_dict = {
                "acc": torchmetrics.Accuracy(task='binary').to(self.args.device),

                "kappa": torchmetrics.CohenKappa(task='binary').to(self.args.device),

                #"prec": torchmetrics.Precision(task = "multiclass",num_classes=self.num_classes, average = None).to(self.args.device),

                #"recall": torchmetrics.Recall(task = "multiclass",num_classes=self.num_classes, average = None).to(self.args.device),

                "f1_macro": torchmetrics.F1Score(task="binary").to(self.args.device),

                "auroc": torchmetrics.AUROC(task="binary").to(self.args.device)

            } 
        elif self.args.task == 'regression':
            classifier_metrics_dict = {

                "mae": torchmetrics.MeanAbsoluteError().to(self.args.device),

                #"expvar": torchmetrics.ExplainedVariance().cuda()#.to(self.args.device),

                "r2": torchmetrics.R2Score().to(self.args.device),

                "pearson": torchmetrics.PearsonCorrCoef().to(self.args.device),

                "spearman": torchmetrics.SpearmanCorrCoef().to(self.args.device)

            }
            if self.args.label=="dep" and self.args.dataset=="shhs2":
                classifier_metrics_dict["5_mae"] = CustomMAE(threshold=5).to(self.args.device)
                classifier_metrics_dict["7_mae"] = CustomMAE(threshold=7).to(self.args.device)

        self.classifier_metrics_dict = classifier_metrics_dict
        
    def fill_metrics(self, raw_predictions, raw_labels):
        if self.args.task == 'regression':
            self.classifier_metrics_dict["r2"].update(raw_predictions, raw_labels)
            self.classifier_metrics_dict["mae"].update(raw_predictions, raw_labels)
            self.classifier_metrics_dict["pearson"].update(raw_predictions, raw_labels.float())
            self.classifier_metrics_dict["spearman"].update(raw_predictions, raw_labels.float())
            #self.classifier_metrics_dict["expvar"].update(raw_predictions, raw_labels)
            if self.args.label == "dep" and self.args.dataset == "shhs2":
                self.classifier_metrics_dict["5_mae"].update(raw_predictions, raw_labels)
                self.classifier_metrics_dict["7_mae"].update(raw_predictions, raw_labels)
        elif self.args.task == 'multiclass':
            #bp()
            predictions, labels = self.multiclass_prediction(raw_predictions, raw_labels)
            self.classifier_metrics_dict["acc"].update(predictions, labels)
            self.classifier_metrics_dict["kappa"].update(predictions, labels)
            #self.classifier_metrics_dict["prec"].update(predictions, labels)
            #self.classifier_metrics_dict["recall"].update(predictions, labels)
            self.classifier_metrics_dict["f1_macro"].update(predictions, labels)
            self.classifier_metrics_dict["f1_c"].update(predictions, labels)
            #bp()
            self.classifier_metrics_dict["auroc"].update(raw_predictions, labels) # need raw_preds for threshold, but need the binary labels
            if self.args.label == "dep":
                #could also just feed the following the argmax'd predictions
                self.classifier_metrics_dict["avg_dep_score"].update(raw_predictions, raw_labels) # needs the raw zung scores to do the calculation
        elif self.args.task == 'binary':
            #bp()
            labels = raw_labels
            raw_predictions = torch.sigmoid(raw_predictions)
            predictions = (raw_predictions > 0.5).squeeze(1)
            
            self.classifier_metrics_dict["acc"].update(predictions, labels)
            self.classifier_metrics_dict["kappa"].update(predictions, labels)
            #self.classifier_metrics_dict["prec"].update(predictions, labels)
            #self.classifier_metrics_dict["recall"].update(predictions, labels)
            self.classifier_metrics_dict["f1_macro"].update(predictions, labels)
            #bp()
            self.classifier_metrics_dict["auroc"].update(raw_predictions, labels) # need raw_preds for threshold, but need the binary labels
            
                
    def compute_and_log_metrics(self, loss, hy_loss=0, classwise_prec_recall=True, classwise_f1=True):
        if self.args.task == 'multiclass':
            #prec = self.classifier_metrics_dict["prec"].compute()
            #rec = self.classifier_metrics_dict["recall"].compute()
            #f1 = self.classifier_metrics_dict["f1_macro"].compute()
            f1_c = self.classifier_metrics_dict["f1_c"].compute()
            
            # "hy_loss": hy_loss,
            # "mae": self.classifier_metrics_dict["mae"].compute(),
            # "expvar": self.classifier_metrics_dict["expvar"].compute(),
            # "r2": self.classifier_metrics_dict["r2"].compute(),
            metrics = {
                
                "total_loss": loss, 
                "acc": self.classifier_metrics_dict["acc"].compute(),
                #"r2": self.classifier_metrics_dict["r2"].compute(),
                #"mae": self.classifier_metrics_dict["mae"].compute(),
                # "kappa": self.classifier_metrics_dict["kappa"].compute(),
                # "neg_precision": neg_prec,
                # "pos_precision": pos_prec,
                # "neg_recall": neg_rec,
                # "pos_recall": pos_rec,
                "f1_macro": self.classifier_metrics_dict["f1_macro"].compute(),
                "auroc": self.classifier_metrics_dict["auroc"].compute()
                
                }
            if classwise_f1:
                    for i in range(self.num_classes):
                        metrics[str(i) + "_f1"] = f1_c[i]
            # if classwise_prec_recall:
            #     for i in range(self.num_classes):
            #         metrics[str(i) + "_precision"] = prec[i]
            #         metrics[str(i) + "_recall"] = rec[i]

            if self.args.label == "dep":
                metrics["avg_dep_score"] = self.classifier_metrics_dict["avg_dep_score"].compute()
        elif self.args.task == 'binary':
            metrics = {
                
                "total_loss": loss, 
                "acc": self.classifier_metrics_dict["acc"].compute(),
                #"r2": self.classifier_metrics_dict["r2"].compute(),
                #"mae": self.classifier_metrics_dict["mae"].compute(),
                # "kappa": self.classifier_metrics_dict["kappa"].compute(),
                # "neg_precision": neg_prec,
                # "pos_precision": pos_prec,
                # "neg_recall": neg_rec,
                # "pos_recall": pos_rec,
                "f1_macro": self.classifier_metrics_dict["f1_macro"].compute(),
                "auroc": self.classifier_metrics_dict["auroc"].compute()
                
                }
            
        elif self.args.task == 'regression':

            metrics = {
                "total_loss": loss,
                "r2": self.classifier_metrics_dict["r2"].compute(),
                "mae": self.classifier_metrics_dict["mae"].compute(),
                "pearson": self.classifier_metrics_dict["pearson"].compute(),
                "spearman": self.classifier_metrics_dict["spearman"].compute()
            }
            if self.args.label=="dep" and self.args.dataset=="shhs2":
                metrics["5_mae"] = self.classifier_metrics_dict["5_mae"].compute()
                metrics["7_mae"] = self.classifier_metrics_dict["7_mae"].compute()

        # self.logger(writer, metrics, phase, epoch)
        return metrics 
    
    def clear_metrics(self):
            for _, val in self.classifier_metrics_dict.items():
                val.reset()

# include mae above 5 and mae above 7