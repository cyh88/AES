from scipy.stats import pearsonr, spearmanr, kendalltau
import logging
import numpy as np
from .my_kappa_calculator import quadratic_weighted_kappa as qwk
from .my_kappa_calculator import linear_weighted_kappa as lwk

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self,get_Data,args,out_dir,dev_x,inputs_dev_ids,inputs_dev_mask,inputs_dev_tokentype,test_x, inputs_test_ids,inputs_test_mask,inputs_test_tokentype,dev_y, test_y,dev_y_org, test_y_org):

        self.dataset = get_Data
        self.prompt_id = args.prompt_id
        self.model = args.model_type
        self.out_dir = out_dir
        self.dev_x, self.inputs_dev_ids, self.inputs_dev_mask, self.inputs_dev_tokentype = dev_x, inputs_dev_ids, inputs_dev_mask, inputs_dev_tokentype
        self.test_x, self.inputs_test_ids, self.inputs_test_mask, self.inputs_test_tokentype = test_x, inputs_test_ids, inputs_test_mask, inputs_test_tokentype
        self.dev_y, self.test_y = dev_y, test_y
        self.dev_y_org, self.test_y_org = dev_y_org, test_y_org
        self.dev_mean = self.dev_y_org.mean()
        self.test_mean = self.test_y_org.mean()
        self.dev_std = self.dev_y_org.std()
        self.test_std = self.test_y_org.std()
        self.best_dev = [-1, -1, -1, -1]
        self.best_test = [-1, -1, -1, -1]
        self.best_dev_epoch = -1
        self.best_test_missed = -1
        self.best_test_missed_epoch = -1
        # self.batch_size = 180
        self.batch_size = args.batch_size
        self.low, self.high = self.dataset.get_score_range(self.prompt_id)
        self.dump_ref_scores()

        self.QWK_best = float("-inf")
        self.LWK_best = float("-inf")
        self.PRS_best = float("-inf")
        self.SPR_best = float("-inf")
        self.Tau_best = float("-inf")
        # 这里进行判断，如果logger.handlers列表为空，则添加，否则，直接去写日志
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())


    def dump_ref_scores(self):
        np.savetxt(self.out_dir + '/preds/dev_ref.txt', self.dev_y_org, fmt='%i')
        np.savetxt(self.out_dir + '/preds/test_ref.txt', self.test_y_org, fmt='%i')

    def dump_predictions(self, dev_pred, test_pred, epoch):
        np.savetxt(self.out_dir + '/preds/dev_pred_' + str(epoch) + '.txt', dev_pred, fmt='%.8f')
        np.savetxt(self.out_dir + '/preds/test_pred_' + str(epoch) + '.txt', test_pred, fmt='%.8f')

    def calc_correl(self, dev_pred, test_pred):
        dev_prs, _ = pearsonr(dev_pred, self.dev_y_org)
        test_prs, _ = pearsonr(test_pred, self.test_y_org)
        dev_spr, _ = spearmanr(dev_pred, self.dev_y_org)
        test_spr, _ = spearmanr(test_pred, self.test_y_org)
        dev_tau, _ = kendalltau(dev_pred, self.dev_y_org)
        test_tau, _ = kendalltau(test_pred, self.test_y_org)
        return dev_prs, test_prs, dev_spr, test_spr, dev_tau, test_tau

    def calc_qwk(self, dev_pred, test_pred):
        # Kappa only supports integer values
        dev_pred_int = np.rint(dev_pred).astype('int32')
        test_pred_int = np.rint(test_pred).astype('int32')
        dev_qwk = qwk(self.dev_y_org, dev_pred_int, self.low, self.high)
        test_qwk = qwk(self.test_y_org, test_pred_int, self.low, self.high)
        dev_lwk = lwk(self.dev_y_org, dev_pred_int, self.low, self.high)
        test_lwk = lwk(self.test_y_org, test_pred_int, self.low, self.high)
        return dev_qwk, test_qwk, dev_lwk, test_lwk

    def evaluate(self, model, epoch, print_info=False):
        logger.info('Evaluating -> dev:')
        self.dev_loss, self.dev_metric = model.evaluate([self.dev_x, self.inputs_dev_ids, self.inputs_dev_mask, self.inputs_dev_tokentype], self.dev_y, batch_size=self.batch_size, verbose=1)
        logger.info('Evaluating -> test:')
        self.test_loss, self.test_metric = model.evaluate([self.test_x, self.inputs_test_ids, self.inputs_test_mask, self.inputs_test_tokentype], self.test_y, batch_size=self.batch_size, verbose=1)
        self.dev_pred = model.predict([self.dev_x, self.inputs_dev_ids, self.inputs_dev_mask, self.inputs_dev_tokentype], batch_size=self.batch_size).squeeze()
        self.test_pred = model.predict([self.test_x, self.inputs_test_ids, self.inputs_test_mask, self.inputs_test_tokentype], batch_size=self.batch_size).squeeze()


        self.dev_pred = self.dataset. convert_to_dataset_friendly_scores(self.dev_pred, self.prompt_id)
        self.test_pred = self.dataset. convert_to_dataset_friendly_scores(self.test_pred, self.prompt_id)


        self.dump_predictions(self.dev_pred, self.test_pred, epoch)
        # print('self.dev_pred=', self.dev_pred)
        self.dev_prs, self.test_prs, self.dev_spr, self.test_spr, self.dev_tau, self.test_tau = self.calc_correl(self.dev_pred, self.test_pred)

        self.dev_qwk, self.test_qwk, self.dev_lwk, self.test_lwk = self.calc_qwk(self.dev_pred, self.test_pred)

        if (self.dev_qwk > self.best_dev[0] and self.dev_qwk > self.best_test[0]) or (self.test_qwk > self.best_dev[0] and self.test_qwk > self.best_test[0] ):
            self.best_dev = [self.dev_qwk, self.dev_lwk, self.dev_prs, self.dev_spr, self.dev_tau]
            self.best_test = [self.test_qwk, self.test_lwk, self.test_prs, self.test_spr, self.test_tau]
            self.best_dev_epoch = epoch


        if self.test_qwk > self.best_test_missed:
            self.best_test_missed = self.test_qwk
            self.best_test_missed_epoch = epoch

        if print_info:
            self.print_info()

    def print_info(self):
        logger.info('[Dev]   loss: %.4f, metric: %.4f, mean: %.3f (%.3f), stdev: %.3f (%.3f)' % (self.dev_loss, self.dev_metric, self.dev_pred.mean(), self.dev_mean, self.dev_pred.std(), self.dev_std))
        logger.info('[Test]  loss: %.4f, metric: %.4f, mean: %.3f (%.3f), stdev: %.3f (%.3f)' % (self.test_loss, self.test_metric, self.test_pred.mean(), self.test_mean, self.test_pred.std(), self.test_std))
        logger.info('[DEV]   QWK:  %.3f, LWK: %.3f, PRS: %.3f, SPR: %.3f, Tau: %.3f (Best @ %i: {{%.3f}}, %.3f, %.3f, %.3f, %.3f)' % (self.dev_qwk, self.dev_lwk, self.dev_prs, self.dev_spr, self.dev_tau, self.best_dev_epoch,self.best_dev[0],self.best_dev[1], self.best_dev[2], self.best_dev[3], self.best_dev[4]))
        logger.info('[TEST]  QWK:  %.3f, LWK: %.3f, PRS: %.3f, SPR: %.3f, Tau: %.3f (Best @ %i: {{%.3f}}, %.3f, %.3f, %.3f, %.3f)' % (self.test_qwk, self.test_lwk, self.test_prs, self.test_spr, self.test_tau, self.best_dev_epoch,self.best_test[0], self.best_test[1], self.best_test[2], self.best_test[3], self.best_test[4]))
        logger.info('--------------------------------------------------------------------------------------------------------------------------')


    def print_final_info(self):
        logger.info('--------------------------------------------------------------------------------------------------------------------------')
        logger.info('Missed @ Epoch %i:' % self.best_test_missed_epoch)
        logger.info('  [TEST] QWK: %.3f' % self.best_test_missed)
        logger.info('Best @ Epoch %i:' % self.best_dev_epoch)
        logger.info('  [DEV]  QWK: %.3f, LWK: %.3f, PRS: %.3f, SPR: %.3f, Tau: %.3f' % (self.best_dev[0], self.best_dev[1], self.best_dev[2], self.best_dev[3], self.best_dev[4]))
        logger.info('  [TEST] QWK: %.3f, LWK: %.3f, PRS: %.3f, SPR: %.3f, Tau: %.3f' % (self.best_test[0], self.best_test[1], self.best_test[2], self.best_test[3], self.best_test[4]))

        self.QWK_best = max(self.best_dev[0],self.best_test[0])


    def get_best_statistics(self):
        # return self.QWK_best,self.LWK_best,self.PRS_best,self.SPR_best,self.Tau_best
        return self.QWK_best