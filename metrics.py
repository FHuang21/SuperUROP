import torchmetrics

class Metrics():
    def __init__(self, args):
        self.args = args 
        
        if self.args.task == 'multiclass':
            self.num_classes = 2
        elif self.args.task == 'regression':
            self.num_classes = 8
        
        self.init_metrics()
        
    def init_metrics(self):
        # if self.args.task == 'multiclass':
            classifier_metrics_dict = {
                "acc": torchmetrics.Accuracy(task='multiclass',num_classes=self.num_classes).to(self.args.device),

                "kappa": torchmetrics.CohenKappa(task='multiclass',num_classes=self.num_classes).to(self.args.device),

                "prec": torchmetrics.Precision(task = "multiclass",num_classes=self.num_classes, average = None).to(self.args.device),

                "recall": torchmetrics.Recall(task = "multiclass",num_classes=self.num_classes, average = None).to(self.args.device),

                "f1": torchmetrics.F1Score(task = 'multiclass',num_classes=self.num_classes).to(self.args.device)
            }
            self.classifier_metrics_dict = classifier_metrics_dict
            
        # elif self.args.task == 'regression':
        #     self.classifier_metrics_dict = {}
        
    def fill_metrics(self, predictions, labels):
                        self.classifier_metrics_dict["acc"](predictions, labels)
                        self.classifier_metrics_dict["kappa"](predictions, labels)
                        self.classifier_metrics_dict["prec"](predictions, labels)
                        self.classifier_metrics_dict["recall"](predictions, labels)
                        self.classifier_metrics_dict["f1"](predictions, labels)
    def compute_and_log_metrics(self, loss):
        # if self.args.task == 'multiclass':
            prec = self.classifier_metrics_dict["prec"].compute()
            rec = self.classifier_metrics_dict["recall"].compute()
            

            metrics = {
                "loss": loss,
                "acc": self.classifier_metrics_dict["acc"].compute(),
                "kappa": self.classifier_metrics_dict["kappa"].compute(),
                # "neg_precision": neg_prec,
                # "pos_precision": pos_prec,
                # "neg_recall": neg_rec,
                # "pos_recall": pos_rec,
                "f1": self.classifier_metrics_dict["f1"].compute(),
                }
            for i in range(self.num_classes):
                metrics[str(i) + "_precision"] = prec[i]
                metrics[str(i) + "_recall"] = rec[i]
                
            # self.logger(writer, metrics, phase, epoch)
            return metrics 
    
    def clear_metrics(self):
            for _, val in self.classifier_metrics_dict.items():
                val.reset()