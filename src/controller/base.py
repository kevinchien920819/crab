import math
from contextlib import nullcontext
from logging import Logger
from pathlib import Path

import numpy as np
import torch
from config import BaseConfig
from controller.eval import compute_eer_from_labels, compute_uar, compute_war
from data import Batch
from torch import Tensor
from tqdm.auto import tqdm

from model.loss import PairwiseGaussianLoss



class Controller:
    def __init__(
            self,
            logger: Logger,
            cfg: BaseConfig,
            model: torch.nn.Module,
            dataloaders: list[torch.utils.data.DataLoader], # [train, val] or [test]
            trial_file: str = None,
            dataset_name: str = None,
        ):
        
        self.cfg            = cfg
        self.logger         = logger
        self.model          = model
        self.dataloaders    = dataloaders
        self.trial_file     = trial_file
        self.dataset_name   = dataset_name
        self.warmup_epochs  = int(cfg.solver.warmup_ratio * cfg.solver.max_epochs)
        self.device         = cfg.general.device
        self.optimizer  = self._setup_optimizer(cfg)
        self.scheduler  = self._setup_scheduler(cfg)
        self.criterions = self._setup_criterions(cfg)
        
        self.is_model_freeze = (cfg.general.freeze != [])
        self.amp_dtype = getattr(cfg.solver, 'amp_dtype', 'fp16')
        if self.amp_dtype not in ['fp16', 'bf16', 'none']:
            raise ValueError(f'Unsupported amp_dtype: {self.amp_dtype}')
        self.amp_enabled = self.amp_dtype != 'none'
        
        self.current_epoch = 0
        self.best_score = 0
        self.iters = 0
    
    def run(self):
        pass
    
    def _setup_optimizer(self, cfg: BaseConfig) -> torch.optim.Optimizer:
        solver_lr = float(cfg.solver.lr[0])

        # Support modular LRs for the refactored 3-model architecture
        if hasattr(cfg.model, "ssl_lr") and "ssl_model" in self.model:
            param_groups = [
                {"params": self.model['ssl_model'].parameters(), "lr": cfg.model.ssl_lr},
                {"params": self.model['text_model'].parameters(), "lr": cfg.model.text_lr},
                {"params": self.model['ser_model'].parameters(), "lr": cfg.model.ser_lr},
            ]
            self.logger.info(
                f"[Controller] Using modular LRs: ssl={cfg.model.ssl_lr}, text={cfg.model.text_lr}, ser={cfg.model.ser_lr}"
            )
        else:
            model_lr = getattr(cfg.model, "lr", None) if hasattr(cfg, "model") else None
            if model_lr is not None:
                model_lr = float(model_lr)
                ssl_params = []
                other_params = []

                for name, param in self.model.named_parameters():
                    if "ssl_model" in name or "text_model" in name:
                        ssl_params.append(param)
                    else:
                        other_params.append(param)

                param_groups = [
                    {"params": ssl_params, "lr": model_lr},
                    {"params": other_params, "lr": solver_lr},
                ]
                self.logger.info(
                    f"[Controller] Using multi-LR: ssl_lr={model_lr}, other_lr={solver_lr}"
                )
            else:
                param_groups = self.model.parameters()
                self.logger.info(f"[Controller] Using single-LR: {solver_lr}")

        if cfg.solver.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                param_groups, lr=solver_lr, weight_decay=float(cfg.solver.weight_decay)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {cfg.solver.optimizer}")
        return optimizer
    
    def _setup_scheduler(self, cfg: BaseConfig) -> torch.optim.lr_scheduler.LRScheduler:
        def div(current_epoch: int) -> float:
            num_lr_segments = len(self.cfg.solver.lr)
            segment_length = self.cfg.solver.max_epochs // num_lr_segments
            segment_index = min(current_epoch // segment_length, num_lr_segments - 1)
            return float(self.cfg.solver.lr[segment_index]) / float(self.cfg.solver.lr[0])
        
        def cosine_warmup(epoch: int) -> float:
            total_epochs = self.cfg.solver.max_epochs
            min_lr_ratio = self.cfg.solver.min_lr_ratio
            
            if epoch < self.warmup_epochs:
                return float(epoch) / float(max(1, self.warmup_epochs))
            
            progress = (epoch - self.warmup_epochs) / float(max(1, total_epochs - self.warmup_epochs))
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        
        if cfg.solver.scheduler == 'div':
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=div
            )
        
        elif cfg.solver.scheduler == 'cosine_warmup':
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=cosine_warmup
            )
        elif cfg.solver.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.solver.max_epochs - self.warmup_epochs
            )
        else:
            raise ValueError(f'Unsupported scheduler: {cfg.solver.scheduler}')
        
        return scheduler
    
    def _calculate_weights(self):
        if not self.dataloaders or self.dataloaders[0] is None:
            return None
        
        train_dataloader = self.dataloaders[0]
        dataset = train_dataloader.dataset
        labels = []
        
        # Accessing private data might be risky, but it's where the labels are preloaded
        # Assuming dataset has .data which is list[Sample]
        if hasattr(dataset, 'data'):
            for sample in dataset.data:
                label = sample.emotion_label if sample.emotion_label is not None else sample.deepfake_label
                if label is not None:
                    labels.append(label)
        
        if not labels:
            return None
            
        labels = np.array(labels)
        classes = np.unique(labels)
        counts = np.bincount(labels)
        
        # Calculate inverse class frequency
        # weight = total_samples / (num_classes * class_count)
        weights = np.zeros_like(counts, dtype=np.float32)
        total = np.sum(counts)
        num_classes = len(classes)
        
        for c in classes:
            if counts[c] > 0:
                weights[c] = total / (num_classes * counts[c])
        
        # Normalize to sum to num_classes (standard way) or just leave it
        # weights = weights * (num_classes / np.sum(weights))
        
        return torch.from_numpy(weights).float().to(self.device)

    def _setup_criterions(self, cfg: BaseConfig) -> list:
        criterions = []
        for loss_name in cfg.solver.criterions.keys():
            loss_cfg = cfg.solver.criterions[loss_name]
            
            if loss_cfg.get('total_weight', 1.0) == 0:
                continue
            
            if loss_name == 'ce_loss':
                criterions.append((
                    loss_name,
                    torch.nn.CrossEntropyLoss(
                        label_smoothing=loss_cfg.get('label_smoothing', 0.0),
                        weight=None,
                    ),
                    loss_cfg['total_weight']
                ))
            elif loss_name == 'wce_loss':
                weight_cfg = loss_cfg.get('weight', 0)
                if isinstance(weight_cfg, list):
                    weights = torch.tensor(weight_cfg).float().to(self.device)
                    self.logger.info(f'[Controller] Using manual wce_loss weights: {weights.tolist()}')
                elif weight_cfg == 0:
                    weights = self._calculate_weights()
                    if weights is not None:
                        self.logger.info(f'[Controller] Calculated wce_loss weights: {weights.tolist()}')
                    else:
                        self.logger.warning('[Controller] Could not calculate weights for wce_loss, using None')
                else:
                    weights = None

                criterions.append((
                    loss_name,
                    torch.nn.CrossEntropyLoss(
                        label_smoothing=loss_cfg.get('label_smoothing', 0.0),
                        weight=weights,
                    ),
                    loss_cfg['total_weight']
                ))

            elif loss_name == 'pg_loss':
                criterions.append((
                    loss_name,
                    PairwiseGaussianLoss(
                        n_classes=loss_cfg['n_classes'],
                        beta=loss_cfg['beta']
                    ),
                    loss_cfg['total_weight']
                ))
            
            # Support for Multi-Positive Contrastive Loss
            elif 'supcon' in loss_name:
                from model.loss import MultiPosConLoss
                criterions.append((
                    loss_name,
                    MultiPosConLoss(
                        temperature=loss_cfg.get('temperature', 0.1)
                    ),
                    loss_cfg['total_weight']
                ))
            
        return criterions
    
    def _cal_loss(self, m_out, b: Batch):
        total_loss: Tensor = torch.tensor(0.0, device=self.device)
        loss_list = []
        
        labels = b.deepfake_labels if b.deepfake_labels is not None else b.emotion_labels
        
        # Define contrastive components if embeddings are present
        embeddings = getattr(m_out, 'embeddings', None)
        contrastive_losses = []
        
        for loss_data in self.criterions:
            loss: Tensor = torch.tensor(0.0, device=self.device)
            loss_name, criterion, weight = loss_data
            
            # 1. Classification Losses
            current_logits = None
            if loss_name == 'ssl_ce_loss':
                current_logits = getattr(m_out, 'ssl_logits', None)
            elif loss_name == 'text_ce_loss':
                current_logits = getattr(m_out, 'text_logits', None)
            elif loss_name in ['ce_loss', 'wce_loss']:
                current_logits = m_out.logits
            
            if current_logits is not None:
                if current_logits.dim() == 3:
                    loss = criterion(current_logits.permute(0, 2, 1), labels)
                else:
                    loss = criterion(current_logits, labels)
                total_loss = total_loss + loss * weight
                loss_list.append(loss.item())
                continue

            # 2. Pairwise Gaussian Loss
            if loss_name == 'pg_loss':
                feature: Tensor = m_out.feature # [B,D]
                dist = PairwiseGaussianLoss.euclidean_dist_all(feature)
                loss = criterion(dist, labels)
                total_loss = total_loss + loss * weight
                loss_list.append(loss.item())
                continue

            # 3. Multimodal Contrastive Losses (Integrated from snippet)
            if 'supcon' in loss_name and embeddings is not None:
                # We expect criterions for audio, text, and fusion if needed, 
                # or we use the same criterion for all as in the snippet.
                
                # Check if this specific criterion is meant for a specific embedding
                # Otherwise, we calculate the combined contrastive loss as per snippet
                l_speech_frame = criterion(embeddings['speech_frame_emb'], labels)
                l_text_frame = criterion(embeddings['text_frame_emb'], labels)
                l_speech_pooled = criterion(embeddings['speech_pooled_emb'], labels)
                l_text_pooled = criterion(embeddings['text_pooled_emb'], labels)
                l_fusion = criterion(embeddings['fusion_emb'], labels)

                # Simple average with multiplier 2.0 as per snippet
                combined_contrastive = 2.0 * (
                    l_speech_frame + l_text_frame + 
                    l_speech_pooled + l_text_pooled + 
                    l_fusion
                ) / 5
                
                total_loss = total_loss + combined_contrastive * weight
                loss_list.append(combined_contrastive.item())
                continue
        
        return total_loss, loss_list
    
    def update_model(self, scaler: torch.amp.GradScaler):
        if scaler is not None:
            scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.solver.max_grad_norm)
        if scaler is not None:
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()
    
    def do_epoch(self, dataloader, scaler: torch.amp.GradScaler = None, backward: bool = True):
        
        true = []
        pred = []      
        fname_list = []
        score_list = []

        if backward:
            self.model.train()
        else:
            self.model.eval()
        
        epoch_loss = 0.0
        epoch_loss_list = [0.0 for _ in self.criterions]
        processed_batches = 0
        last_batch_idx = -1
        
        bar = tqdm(dataloader, total=len(dataloader), unit='batch', desc='Training' if backward else 'Testing')
        amp_dtype = torch.float16 if self.amp_dtype == 'fp16' else torch.bfloat16
        for b_idx, b in enumerate(bar):
            last_batch_idx = b_idx
            b: Batch
            b.to(self.device, non_blocking=self.cfg.dataloader.non_blocking_transfer)
            
            # forward
            with torch.no_grad() if not backward else torch.enable_grad():
                amp_ctx = torch.amp.autocast(self.device, dtype=amp_dtype) if self.amp_enabled else nullcontext()
                with amp_ctx:
                    m_out = self.model(b)
                    if self.cfg.general.produce_evaluation_file and not backward:
                        batch_score = (m_out.logits[:, 0, 0]).data.cpu().numpy().ravel()
                        # add outputs
                        fname_list.extend(Path(p).stem for p in b.path)
                        score_list.extend(batch_score.tolist())
                    
                    batch_loss, batch_loss_list = self._cal_loss(m_out, b)
            
            if not torch.isfinite(batch_loss):
                self.logger.warning('[Trainer] Skip NaN/Inf batch at idx %d', b_idx)
                if backward:
                    self.optimizer.zero_grad()
                continue
            
            for i in range(len(epoch_loss_list)):
                epoch_loss_list[i] += batch_loss_list[i]
            processed_batches += 1
            
            if backward:
                scaled_loss = batch_loss / self.cfg.solver.iters_to_accumulate
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                
                if (b_idx + 1) % self.cfg.solver.iters_to_accumulate == 0:
                    self.update_model(scaler)
            
            epoch_loss += batch_loss.item()
            
            bar.set_postfix({'batch_loss': batch_loss.item()})
            # todo: remove mask it is useless in deepfake detection
            # if self.dataset_name in ['ASVspoof2019_LA', 'ASVspoof2021_LA', 'ASVspoof2021_DF', 'ASVspoof5']:
            #     mask = (b.label != -100)
            # if self.dataset_name in ['MELD', 'IEMOCAP', 'MSP_Podcast']:
            #     mask = (b.emotion_label != -100)
            # logits = m_out.logits
            # score = torch.softmax(logits, dim=-1)[..., 1]
            # true.append(b.label[mask].detach().cpu())
            # pred.append(score[mask].detach().cpu())
            logits = m_out.logits
            if self.dataset_name in ['MELD', 'IEMOCAP', 'MSP_Podcast']:
                true.append(b.emotion_labels.detach().cpu())
                pred.append(logits.argmax(dim=-1).detach().cpu())
            else:
                # Assuming deepfake detection
                if b.deepfake_labels is not None:
                    true.append(b.deepfake_labels.detach().cpu())
                else:
                    true.append(b.emotion_labels.detach().cpu())
                
                score = torch.softmax(logits, dim=-1)[..., 1]
                pred.append(score.detach().cpu())
            
        if backward and last_batch_idx >= 0 and (last_batch_idx + 1) % self.cfg.solver.iters_to_accumulate != 0:
            self.update_model(scaler)
        
        epoch_loss = epoch_loss / processed_batches
        epoch_loss_list = [l / processed_batches for l in epoch_loss_list]
        
        if self.cfg.general.produce_evaluation_file and not backward and self.dataset_name in ['ASVspoof2019_LA', 'ASVspoof2021_LA', 'ASVspoof2021_DF', 'ASVspoof5']:
            assert self.trial_file is not None
            
            # Build trial info dictionary for fast lookup by utt_id
            trial_info = {}
            with open(self.trial_file, "r") as f_trl:
                for line in f_trl:
                    parts = line.strip().split()
                    if not parts: continue
                    
                    if self.dataset_name == 'ASVspoof2019_LA' or self.dataset_name == 'ASVspoof5':
                        spk_id, utt_id, _, _, key = parts
                    elif self.dataset_name == 'ASVspoof2021_LA':
                        spk_id, utt_id, _, _, _, key, _, _ = parts
                    elif self.dataset_name == 'ASVspoof2021_DF':
                        spk_id = parts[0]
                        utt_id = parts[1]
                        key = parts[5]
                    else:
                        # Fallback
                        spk_id = parts[0]
                        utt_id = parts[1]
                        key = parts[-1]
                    
                    trial_info[utt_id] = (spk_id, key)

            with open(self.cfg.general.work_dir + f"/evaluation_scores.txt", "w") as fh:
                for fn, sco in zip(fname_list, score_list):
                    if fn in trial_info:
                        spk_id, key = trial_info[fn]
                        fh.write("{} {} {} {}\n".format(spk_id, fn, sco, key))
                    else:
                        self.logger.warning(f'[Controller] {fn} not found in trial file, skipping.')

        
        return true, pred, epoch_loss, epoch_loss_list

    def cal_eer(self, true: list[torch.Tensor], pred: list[torch.Tensor]) -> float:
        if not true or not pred:
            self.logger.warning('Skip EER calculation because no predictions were produced')
            return float('nan')
        
        true = torch.cat(true).view(-1).numpy()
        pred = torch.cat(pred).view(-1).numpy()
        
        try:
            return compute_eer_from_labels(true, pred)
        except ValueError as exc:
            self.logger.warning('Skip EER calculation because %s', exc)
            return float('nan')
        
    def cal_uar(self, true: list[torch.Tensor], pred: list[torch.Tensor]) -> float:
        if not true or not pred:
            self.logger.warning('Skip UAR calculation because no predictions were produced')
            return float('nan')
        
        true = torch.cat(true).view(-1).numpy()
        pred = torch.cat(pred).view(-1).numpy()
        
        try:
            return compute_uar(true, pred)
        except Exception as exc:
            self.logger.warning('Skip UAR calculation because %s', exc)
            return float('nan')

    def cal_war(self, true: list[torch.Tensor], pred: list[torch.Tensor]) -> float:
        if not true or not pred:
            self.logger.warning('Skip WAR calculation because no predictions were produced')
            return float('nan')
        
        true = torch.cat(true).view(-1).numpy()
        pred = torch.cat(pred).view(-1).numpy()
        
        try:
            return compute_war(true, pred)
        except Exception as exc:
            self.logger.warning('Skip WAR calculation because %s', exc)
            return float('nan')