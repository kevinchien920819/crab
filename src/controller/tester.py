from logging import Logger

import os
import numpy as np
import torch
from config import BaseConfig
from wandb import Run

from evaluation_metric.calculate_modules import calculate_CLLR, compute_actDCF, compute_eer, compute_mindcf

from .base import Controller

class Tester(Controller):
    def __init__(
            self,
            logger: Logger,
            cfg: BaseConfig,
            wandb_run: Run,
            model: torch.nn.Module,
            dataloaders: list[torch.utils.data.DataLoader],
            trial_file: str,
            dataset_name: str,
            eval_split: str = 'test',
        ):

        """Initialize an evaluation controller for one dataset split."""
        super(Tester, self).__init__(
            logger,
            cfg,
            model,
            dataloaders,
            trial_file,
            dataset_name
        )
        self.wandb_run = wandb_run
        self.wandb_log = {}
        self.eval_split = eval_split


    def run(self) -> tuple[dict, float]:
        """Evaluate the configured dataloader and return dataset-appropriate metrics."""
        test_dataloader = self.dataloaders[0]
        log_prefix = self.eval_split or 'test'

        test_true, test_pred, test_loss, test_loss_list = self.do_epoch(test_dataloader, backward=False)
        self.logger.info(f'[Tester:{log_prefix}] Test Loss components: ' + ', '.join([f'loss_{i}: {test_loss_list[i]:.4f}' for i in range(len(test_loss_list))]))
        self.logger.info(f'[Tester:{log_prefix}] Testing loss: {test_loss:.4f}')

        if self.dataset_name in ['MELD', 'IEMOCAP', 'MSP_Podcast']:
            test_war = self.cal_war(test_true, test_pred)
            test_uar = self.cal_uar(test_true, test_pred)
            self.logger.info(f'[Tester:{log_prefix}] Testing WAR: {test_war:.4f}, Testing UAR: {test_uar:.4f}')
        else:
            test_eer = self.cal_eer(test_true, test_pred)
            self.logger.info(f'[Tester:{log_prefix}] Testing EER: {test_eer:.4f}')


        if self.cfg.wandb.enable:
            self.wandb_log.update({f'{log_prefix}_loss': test_loss})
            self.wandb_log.update({f'{log_prefix}_loss_components_{i}': test_loss_list[i] for i in range(len(test_loss_list))})

            if self.dataset_name  in ['MELD', 'IEMOCAP', 'MSP_Podcast']:
                self.wandb_log.update({f'{log_prefix}_war': test_war, f'{log_prefix}_uar': test_uar})
            else:
                self.wandb_log.update({f'{log_prefix}_eer': test_eer})
            self.wandb_run.log(self.wandb_log)
        if self.dataset_name  in ['ASVspoof2019_LA', 'ASVspoof2021_LA', 'ASVspoof2021_DF', 'ASVspoof5']:
            return test_eer, test_loss
        if self.dataset_name in ['MELD', 'IEMOCAP', 'MSP_Podcast']:
            return test_war, test_uar, test_loss


    # TODO get min-tDCF
    def _calculate_minDCF_EER_CLLR_actDCF(
            self,
            cm_scores_file,
            output_file,
            printout=True
        ):
        # Evaluation metrics for Phase 1
        # Primary metrics: min DCF,
        # Secondary metrics: EER, CLLR

        """Calculate ASVspoof CM metrics from an evaluation score file."""
        Pspoof = 0.05
        dcf_cost_model = {
            'Pspoof': Pspoof,  # Prior probability of a spoofing attack
            'Cmiss': 1,  # Cost of CM system falsely rejecting target speaker
            'Cfa' : 10, # Cost of CM system falsely accepting nontarget speaker
        }


        # Load CM scores
        cm_data = np.genfromtxt(cm_scores_file, dtype=str)
        cm_keys = cm_data[:, 3]
        cm_scores = cm_data[:, 2].astype(np.float64)

        # Extract bona fide (real human) and spoof scores from the CM scores
        bona_cm = cm_scores[cm_keys == 'bonafide']
        spoof_cm = cm_scores[cm_keys == 'spoof']

        # EERs of the standalone systems and fix ASV operating point to EER threshold
        eer_cm, frr, far, thresholds, _ = compute_eer(bona_cm, spoof_cm)#[0]
        cllr_cm = calculate_CLLR(bona_cm, spoof_cm)
        minDCF_cm, _ = compute_mindcf(frr, far, thresholds, Pspoof, dcf_cost_model['Cmiss'], dcf_cost_model['Cfa'])
        actDCF, _ = compute_actDCF(bona_cm, spoof_cm, Pspoof, dcf_cost_model['Cmiss'], dcf_cost_model['Cfa'])

        if printout:
            with open(output_file, "w") as f_res:
                f_res.write('\nCM SYSTEM\n')
                f_res.write('\tmin DCF \t\t= {} % '
                            '(min DCF for countermeasure)\n'.format(
                                minDCF_cm))
                f_res.write('\tEER\t\t= {:8.9f} % '
                            '(EER for countermeasure)\n'.format(
                                eer_cm * 100))
                f_res.write('\tCLLR\t\t= {:8.9f} % '
                            '(CLLR for countermeasure)\n'.format(
                                cllr_cm * 100))
                f_res.write('\tactDCF\t\t= {:} '
                            '(actual DCF)\n'.format(
                                actDCF))
            os.system(f"cat {output_file}")

        return minDCF_cm, eer_cm, cllr_cm, actDCF
