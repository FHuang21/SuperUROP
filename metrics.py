import torchmetrics
import torch 

class Metrics():
    def __init__(self, args):
        self.args = args 
        
        if self.args.task == 'multiclass':
            self.num_classes = 2
        elif self.args.target == 'hy':
            self.num_classes = 8
        elif self.args.target == 'age':
            self.num_classes = 2
            
        if self.args.target == 'hy':
            self.transform_pred = self.nearest_hy_score
        elif self.args.target == 'age':
            self.transform_pred = self.age_prediction
        elif self.args.task == 'multiclass':
            self.transform_pred = self.multiclass_prediction
            
        self.init_metrics()
    
    def multiclass_prediction(self, predictions, labels):
        predictions = torch.argmax(predictions,dim=1)
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

            # if self.args.task == 'multiclass':
            classifier_metrics_dict = {
                "acc": torchmetrics.Accuracy(task='multiclass',num_classes=self.num_classes).cuda(),#.to(self.args.device),

                "kappa": torchmetrics.CohenKappa(task='multiclass',num_classes=self.num_classes).cuda(),#.to(self.args.device),

                "prec": torchmetrics.Precision(task = "multiclass",num_classes=self.num_classes, average = None).cuda(),#.to(self.args.device),

                "recall": torchmetrics.Recall(task = "multiclass",num_classes=self.num_classes, average = None).cuda(),#.to(self.args.device),

                "f1": torchmetrics.F1Score(task = 'multiclass',num_classes=self.num_classes).cuda()#.to(self.args.device)
            }
            if self.args.task == 'regression':
                classifier_metrics_dict["mae"] = torchmetrics.MeanAbsoluteError().cuda()#.to(self.args.device)
                classifier_metrics_dict["expvar"] = torchmetrics.ExplainedVariance().cuda()#.to(self.args.device)
                classifier_metrics_dict["r2"] = torchmetrics.R2Score().cuda()#.to(self.args.device)
            self.classifier_metrics_dict = classifier_metrics_dict
            
        
    def fill_metrics(self, raw_predictions, raw_labels):
        if self.args.task == 'regression':
            self.classifier_metrics_dict["r2"].update(raw_predictions, raw_labels)
            self.classifier_metrics_dict["mae"].update(raw_predictions, raw_labels)
            self.classifier_metrics_dict["expvar"].update(raw_predictions, raw_labels)
        
        predictions, labels = self.transform_pred(raw_predictions, raw_labels)
        self.classifier_metrics_dict["acc"].update(predictions, labels)
        self.classifier_metrics_dict["kappa"].update(predictions, labels)
        self.classifier_metrics_dict["prec"].update(predictions, labels)
        self.classifier_metrics_dict["recall"].update(predictions, labels)
        self.classifier_metrics_dict["f1"].update(predictions, labels)
                        
    def compute_and_log_metrics(self, loss, hy_loss=0, classwise_prec_recall=True):
        # if self.args.task == 'multiclass':
            prec = self.classifier_metrics_dict["prec"].compute()
            rec = self.classifier_metrics_dict["recall"].compute()
            
            # "hy_loss": hy_loss,
            # "mae": self.classifier_metrics_dict["mae"].compute(),
            # "expvar": self.classifier_metrics_dict["expvar"].compute(),
            # "r2": self.classifier_metrics_dict["r2"].compute(),
            metrics = {
                
                "total_loss": loss, 
                "acc": self.classifier_metrics_dict["acc"].compute(),
                # "kappa": self.classifier_metrics_dict["kappa"].compute(),
                # "neg_precision": neg_prec,
                # "pos_precision": pos_prec,
                # "neg_recall": neg_rec,
                # "pos_recall": pos_rec,
                
                "f1": self.classifier_metrics_dict["f1"].compute(),
                }
            if classwise_prec_recall:
                for i in range(self.num_classes):
                    metrics[str(i) + "_precision"] = prec[i]
                    metrics[str(i) + "_recall"] = rec[i]
                
            # self.logger(writer, metrics, phase, epoch)
            return metrics 
    
    def clear_metrics(self):
            for _, val in self.classifier_metrics_dict.items():
                val.reset()