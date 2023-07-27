import torchmetrics
import torchmetrics.classification
import torch
from ipdb import set_trace as bp

# class AvgDepScore(torchmetrics.Metric):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.num_classes = num_classes
#         self.pred_label_list = []
#         self.actual_label_list = []

#     def update(self, raw_predictions, raw_labels): # all raw labels
#         self.pred_label_list += raw_predictions # append each element to list, not append a list inside the list
#         self.actual_label_list += raw_labels
    
#     def compute(self):

# class AvgDepScore(torchmetrics.Metric): 
#     def __init__(self, threshold, dist_sync_on_step=False): # FIXME : threshold
#         super().__init__()
#         self.add_state("zung_scores", default=[], dist_reduce_fx="cat")
#         self.threshold = threshold

#     def update(self, raw_predictions: torch.tensor, raw_labels: torch.tensor):
#         #self.pred_classes = (raw_predictions >= self.threshold).int()
#         self.zung_scores.extend(raw_predictions)

#     def compute(self):
#         positive_scores = [zung_score for zung_score in self.zung_scores if zung_score >= self.threshold]
#         return sum(positive_scores) / len(positive_scores)

#     def reset(self):
#         self.zung_scores = [] # reset to empty list

class AvgDepScore(torchmetrics.Metric): 
    def __init__(self, threshold, dist_sync_on_step=False): # decide threshold on instantiation
        super().__init__()
        self.add_state("zung_scores", default=[], dist_reduce_fx="cat")
        self.threshold = threshold

    def update(self, raw_predictions, raw_labels: torch.tensor):
        #bp()
        self.pred_classes = torch.argmax(raw_predictions, dim=1)
        #threshd_labels = (raw_labels >= self.threshold).int()

        self.zung_scores.extend(self.pred_classes * raw_labels)

    def compute(self):
        #bp()
        positive_scores = [zung_score.item() for zung_score in self.zung_scores if zung_score.item() >= self.threshold]
        if len(positive_scores) == 0:
            return 0 # no positive examples were predicted i guess???
        else:
            return sum(positive_scores) / len(positive_scores)

    def reset(self):
        self.zung_scores = [] # reset to empty list

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
            self.num_classes = 2 # will still threshold regression output to get binary depression classification metrics
            if self.args.dataset == "wsc":
                self.threshold = 36
            elif self.args.dataset == "shhs2":
                self.threshold = 5
            
        if self.args.target == 'hy':
            self.transform_pred = self.nearest_hy_score
        elif self.args.target == 'age':
            self.transform_pred = self.age_prediction
        elif self.args.task == 'multiclass':
            self.transform_pred = self.multiclass_prediction
        elif self.args.task == 'regression':
            self.transform_pred = self.regression_prediction
            
        self.init_metrics()
    
    def multiclass_prediction(self, predictions, labels):
        predictions = torch.argmax(predictions,dim=1)
        labels = (labels >= self.threshold).int()
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
        
        #if self.args.task == 'multiclass':
        classifier_metrics_dict = {
            "acc": torchmetrics.Accuracy(task='multiclass',num_classes=self.num_classes).to(self.args.device),

            "kappa": torchmetrics.CohenKappa(task='multiclass',num_classes=self.num_classes).to(self.args.device),

            #"prec": torchmetrics.Precision(task = "multiclass",num_classes=self.num_classes, average = None).to(self.args.device),

            #"recall": torchmetrics.Recall(task = "multiclass",num_classes=self.num_classes, average = None).to(self.args.device),

            "f1_macro": torchmetrics.F1Score(task = "multiclass", num_classes=self.num_classes, average = "macro").to(self.args.device),

            "f1_c": torchmetrics.classification.MulticlassF1Score(num_classes=self.num_classes, average = None).to(self.args.device),

            "avg_dep_score": AvgDepScore(threshold=self.threshold).to(self.args.device)

        } 

        if self.args.task == 'regression':
            classifier_metrics_dict["mae"] = torchmetrics.MeanAbsoluteError().to(self.args.device)
            #classifier_metrics_dict["expvar"] = torchmetrics.ExplainedVariance().cuda()#.to(self.args.device)
            classifier_metrics_dict["r2"] = torchmetrics.R2Score().to(self.args.device)

        self.classifier_metrics_dict = classifier_metrics_dict
        
    def fill_metrics(self, raw_predictions, raw_labels):
        if self.args.task == 'regression':
            self.classifier_metrics_dict["r2"].update(raw_predictions, raw_labels)
            self.classifier_metrics_dict["mae"].update(raw_predictions, raw_labels)
            #self.classifier_metrics_dict["expvar"].update(raw_predictions, raw_labels)
        #elif self.args.task == 'multiclass':
        #bp()
        predictions, labels = self.multiclass_prediction(raw_predictions, raw_labels)
        self.classifier_metrics_dict["acc"].update(predictions, labels)
        self.classifier_metrics_dict["kappa"].update(predictions, labels)
        #self.classifier_metrics_dict["prec"].update(predictions, labels)
        #self.classifier_metrics_dict["recall"].update(predictions, labels)
        self.classifier_metrics_dict["f1_macro"].update(predictions, labels)
        self.classifier_metrics_dict["f1_c"].update(predictions, labels)
        if self.args.label == "dep":
            self.classifier_metrics_dict["avg_dep_score"].update(raw_predictions, raw_labels) # needs the raw zung scores to do the calculation
                        
    def compute_and_log_metrics(self, loss, hy_loss=0, classwise_prec_recall=True, classwise_f1=True):
        #if self.args.task == 'multiclass':
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
            "f1_macro": self.classifier_metrics_dict["f1_macro"].compute()
            
            }
        if classwise_f1:
                for i in range(self.num_classes):
                    metrics[str(i) + "_f1"] = f1_c[i]
        # if classwise_prec_recall:
        #     for i in range(self.num_classes):
        #         metrics[str(i) + "_precision"] = prec[i]
        #         metrics[str(i) + "_recall"] = rec[i]

        if self.args.task == 'regression':
            metrics["r2"] = self.classifier_metrics_dict["r2"].compute()
            metrics["mae"] = self.classifier_metrics_dict["mae"].compute()

        if self.args.label == "dep":
            metrics["avg_dep_score"] = self.classifier_metrics_dict["avg_dep_score"].compute()

        # self.logger(writer, metrics, phase, epoch)
        return metrics 
    
    def clear_metrics(self):
            for _, val in self.classifier_metrics_dict.items():
                val.reset()