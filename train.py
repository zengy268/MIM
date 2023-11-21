import os
import torch.nn
from utils.data_preparation import set_up_data_loader
from utils.utils import mkdirs, set_random_seed, Logger
from utils.loss import Loss_fct, Loss_penalty, Loss_intra, Loss_inter, Adjuster
from global_configs import config
from modules.model_github import MIM
from datetime import datetime
from timeit import default_timer as timer
from torch import nn
from torch.optim import Adam
from itertools import chain
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
import warnings
import wandb
import time

wandb.init(mode='disabled')
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = config.device
device = 'cuda'

def log_write(log):
    log.open(config.logs + config.dataset + '_nomask_log.txt', mode = 'a')
    log.write("\n--------------------------------------- [START %s] %s\n\n" % (
        datetime.now().strftime('%Y-%M-%D %H:%M:%S'), '-' * 51))
    log.write("train_batch_size: " + str(config.train_batch_size) + " ,  " +
              "dev_batch_size: " + str(config.dev_batch_size) + " ,  " +
              "test_batch_size: " + str(config.dev_batch_size) + " ,  " +
              "n_epochs: " + str(config.n_epochs) + " ,  " +
              "feature_dimension: " + str(config.d_l) + '\n'
              )
    log.write('** Start Training! **\n')
    log.write(
        '-----------|----------------- CURRENT TEST -----------------|----------------- CURRENT BEST -----------------|-------- LOSS--------|----------|\n')
    log.write(
        '   EPOCH   |     ACC     ACC7      MAE      F1      CORR    |     ACC     ACC7      MAE      F1      CORR    |   TRAIN    VALID    |   TIME   |\n')
    log.write(
        '-----------------------------------------------------------------------------------------------------------------------------------------------\n')


def loss_kd(teacher, student1, student2, alpha_l):
    soft_loss = nn.KLDivLoss(reduction='batchmean') # (bs, dim)
    temp = 10
    temp = temp * alpha_l
    KD_loss = (soft_loss(F.log_softmax(student1, dim=1), F.softmax(teacher, dim=1)) + soft_loss(F.log_softmax(student2, dim=1), F.softmax(teacher, dim=1)))*temp*temp
    return KD_loss * 0.5
    

class Train():
    def __init__(self,train_dataloader, validation_dataloader, test_data_loader):
        # dataloader
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_data_loader = test_data_loader
        # network
        self.model = MIM().to(device)
        # optimizer
        self.optimizer_model =  Adam(self.model.parameters(), lr=config.lr)
        self.optimizer_encoder = Adam(chain(self.model.l_encoder.parameters(),
                                    self.model.a_encoder.parameters(),
                                    self.model.v_encoder.parameters()), lr=config.lr)
        self.optimizer_UPN = Adam(chain(self.model.l_mask.parameters(),
                                    self.model.a_mask.parameters(),
                                    self.model.v_mask.parameters()), lr=config.lr)

        # criterion
        self.loss_fct = Loss_fct()
        self.loss_penalty = Loss_penalty()
        self.loss_intra_KGS = Loss_intra()
        self.loss_inter_KGS = Loss_inter()
        self.weight = 1

    def train_one_epoch(self):
        self.model.train()

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(self.train_dataloader):
            batch = tuple(t.cuda() for t in batch)
            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)

            ##################  reperesentation learning ####################
            # unimodal encoding # (bs, dim) & (ts, bs, dim)
            unmasked_l, output_l = self.model.l_encoder(
                input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,)
            unmasked_a, output_a = self.model.a_encoder(acoustic)
            unmasked_v, output_v = self.model.v_encoder(visual)

            # unimodal purifying network
            p_w = 0.1
            masked_l, penalty_l = self.model.l_mask(output_l)
            masked_a, penalty_a = self.model.a_mask(output_a)
            masked_v, penalty_v = self.model.v_mask(output_v)
            loss_penalty = torch.mean(penalty_l + penalty_a + penalty_v) * p_w

            #########################  KGS ###########################
            # unimodal cls
            outputl_unmasked = self.model.l_classifier(unmasked_l)
            outputa_unmasked = self.model.a_classifier(unmasked_a)
            outputv_unmasked = self.model.v_classifier(unmasked_v)
            outputl_masked = self.model.l_classifier(masked_l)
            outputa_masked = self.model.a_classifier(masked_a)
            outputv_masked = self.model.v_classifier(masked_v)

            loss_l_unmasked = self.loss_fct(outputl_unmasked.view(-1), label_ids.view(-1))
            loss_a_unmasked = self.loss_fct(outputa_unmasked.view(-1), label_ids.view(-1))
            loss_v_unmasked = self.loss_fct(outputv_unmasked.view(-1), label_ids.view(-1))
            loss_l_masked = self.loss_fct(outputl_masked.view(-1), label_ids.view(-1))
            loss_a_masked = self.loss_fct(outputa_masked.view(-1), label_ids.view(-1))
            loss_v_masked = self.loss_fct(outputv_masked.view(-1), label_ids.view(-1))
            loss_unmasked = torch.mean(loss_l_unmasked + loss_a_unmasked + loss_v_unmasked) * self.weight
            loss_masked = torch.mean(loss_l_masked + loss_a_masked + loss_v_masked) * self.weight

            # intra knowledge transfer
            loss_l_intra = self.loss_intra_KGS(loss_l_masked, loss_l_unmasked, outputl_masked.view(-1), outputl_unmasked.view(-1))
            loss_a_intra = self.loss_intra_KGS(loss_a_masked, loss_a_unmasked, outputa_masked.view(-1), outputa_unmasked.view(-1))
            loss_v_intra = self.loss_intra_KGS(loss_v_masked, loss_v_unmasked, outputv_masked.view(-1), outputv_unmasked.view(-1))
            loss_intra = torch.mean(loss_l_intra+loss_a_intra+loss_v_intra)
            
            # # inter knowledge transfer
            alpha_l, alpha_a, alpha_v = Adjuster(loss_l_masked, loss_a_masked, loss_v_masked)

            loss_l_inter, loss_a_inter, loss_v_inter = self.loss_inter_KGS(loss_l_masked, loss_a_masked, loss_v_masked, masked_l, masked_a, masked_v, alpha_l, alpha_a, alpha_v)
            loss_inter = torch.mean(loss_l_inter + loss_a_inter + loss_v_inter)  * self.weight

            #########################  Fusion and Inference ###########################
            # multimodal fusion and inference
            logits = self.model.fusion(masked_l, masked_a, masked_v)
            loss_cls = self.loss_fct(logits.view(-1), label_ids.view(-1))

            #########################  Model Updating ###########################
            self.optimizer_UPN.zero_grad()
            loss_penalty.backward(retain_graph=True)
            loss_masked.backward(retain_graph=True)
            self.optimizer_UPN.step()

            self.optimizer_model.zero_grad()
            loss_unmasked.backward(retain_graph=True)
            loss_inter.backward(retain_graph = True)
            loss_intra.backward(retain_graph = True) # should only update encoder/UPN
            loss_cls.backward()
            self.optimizer_model.step()

            tr_loss += loss_cls.item()
            nb_tr_steps += 1

        return tr_loss / nb_tr_steps


    def eval_one_epoch(self):
        self.model.eval()

        dev_loss = 0
        nb_dev_examples, nb_dev_steps = 0, 0

        with torch.no_grad():
            for step, batch in enumerate(self.validation_dataloader):
                batch = tuple(t.cuda() for t in batch)

                input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
                visual = torch.squeeze(visual, 1)
                acoustic = torch.squeeze(acoustic, 1)

                # unimodal encoding
                unmasked_l, output_l = self.model.l_encoder(
                    input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask, )
                unmasked_a, output_a = self.model.a_encoder(acoustic)
                unmasked_v, output_v = self.model.v_encoder(visual)

                # unimodal purifying network
                masked_l, _ = self.model.l_mask(output_l)
                masked_a, _ = self.model.a_mask(output_a)
                masked_v, _ = self.model.v_mask(output_v)

                # final output
                logits = self.model.fusion(masked_l, masked_a, masked_v)
                loss = self.loss_fct(logits.view(-1), label_ids.view(-1))

                dev_loss += loss.item()
                nb_dev_steps += 1

        return dev_loss / nb_dev_steps


    def test_one_epoch(self):
        self.model.eval()

        preds = []
        labels = []

        with torch.no_grad():
            for batch in self.test_data_loader:
                batch = tuple(t.cuda() for t in batch)

                input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
                visual = torch.squeeze(visual, 1)
                acoustic = torch.squeeze(acoustic, 1)

                # unimodal encoding
                unmasked_l, output_l = self.model.l_encoder(
                    input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask, )
                unmasked_a, output_a = self.model.a_encoder(acoustic)
                unmasked_v, output_v = self.model.v_encoder(visual)

                # unimodal purifying network
                masked_l, _ = self.model.l_mask(output_l)
                masked_a, _ = self.model.a_mask(output_a)
                masked_v, _ = self.model.v_mask(output_v)

                # final output
                logits = self.model.fusion(masked_l, masked_a, masked_v)
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.detach().cpu().numpy()

                logits = np.squeeze(logits).tolist()
                label_ids = np.squeeze(label_ids).tolist()

                preds.extend(logits)
                labels.extend(label_ids)

            preds = np.array(preds)
            labels = np.array(labels)

        return preds, labels


    def multiclass_acc(self, preds, truths):
        """
        Compute the multiclass accuracy w.r.t. groundtruth

        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


    def test_score(self, use_zero=False):

        preds, y_test = self.test_one_epoch()
        non_zeros = np.array(
            [i for i, e in enumerate(y_test) if e != 0 or use_zero])

        test_preds_a7 = np.clip(preds, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(y_test, a_min=-3., a_max=3.)
        mult_a7 = self.multiclass_acc(test_preds_a7, test_truth_a7)

        preds = preds[non_zeros]
        y_test = y_test[non_zeros]
        mae = np.mean(np.absolute(preds - y_test))
        corr = np.corrcoef(preds, y_test)[0][1]

        preds = preds >= 0
        y_test = y_test >= 0
        f_score = f1_score(y_test, preds, average="weighted")
        acc = accuracy_score(y_test, preds)

        return acc, mae, corr, f_score, mult_a7


    def process(self):
        valid_losses = []
        test_accuracies = []
        best_loss = 10
        log = Logger()
        log_write(log)
        for epoch_i in range(int(config.n_epochs)):
            train_loss = self.train_one_epoch()
            valid_loss = self.eval_one_epoch()
            test_acc, test_mae, test_corr, test_f_score, test_acc7 = self.test_score()
            valid_losses.append(valid_loss)
            test_accuracies.append(test_acc)

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_acc = test_acc
                best_mae = test_mae
                best_corr = test_corr
                best_f_score = test_f_score
                best_acc_7 = test_acc7
                
            log.write(
                '   %2d/%d   |   %.4f   %.4f   %.4f   %.4f   %.4f   |   %.4f   %.4f   %.4f   %.4f   %.4f   |   %.4f   %.4f   |'
                % (
                    epoch_i, config.n_epochs,
                    test_acc, test_acc7, test_mae, test_f_score, test_corr,
                    best_acc, best_acc_7, best_mae, best_f_score, best_corr,
                    train_loss, valid_loss))
            log.write('\n')
            time.sleep(0.01)

def main():

    # logs setting
    # mkdirs(config.checkpoint_path, config.best_model_path, config.logs)
    set_random_seed(config.seed)

    # load data
    (train_data_loader, dev_data_loader, test_data_loader, num_train_optimization_steps) = set_up_data_loader(
        config.dataset, config.train_batch_size, config.dev_batch_size, config.test_batch_size, config.n_epochs)

    # model training
    Train(train_data_loader, dev_data_loader, test_data_loader).process()



if __name__ == '__main__':
    main()
