import sys
sys.path.append('../')

from loguru import logger
from argparse import Namespace
import shutil
from abc import ABC, abstractmethod
from abstract_trainer import AbstractTrainer
from models.lassi_encoder import LASSIEncoder
from metrics import ClassificationAndFairnessMetrics
from torchtext.datasets import MNLI
from nltk.tokenize import word_tokenize
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import utils

params = Namespace(adv_loss_weight=0.0, attr_vectors_dir='attr_vectors_avg_diff', batch_size=64, certification_batch_size=32, classify_attributes=['Age_bin'], cls_alpha=0.001, cls_layers=[], cls_loss_weight=1.0, cls_n=100000, cls_n0=2000, cls_sigma=None, cls_sigmas=[5.0], data_augmentation=False, dataset='glow_fairface_latent_lmdb', delta=0.0, enc_alpha=0.01, enc_n=10000, enc_n0=10000, enc_sigma=0.325, encoder_hidden_layers=[2048, 1024], encoder_normalize_output=True, encoder_type='linear', encoder_use_bn=True, epochs=10, fair_classifier_data_augmentation='random_noise', fair_classifier_name=None, fair_encoder_name=None, gen_model_name='glow_fairface', gen_model_type='Glow', glow_affine=False, glow_n_block=4, glow_n_flow=32, glow_no_lu=False, image_size=64, input_representation='latent', lr=0.001, n_bits=5, num_workers=4, parallel=False, perform_endpoints_analysis=False, perturb='Black', perturb_epsilon=0.5, random_attack_num_samples=10, recon_decoder_layers=[], recon_decoder_type='linear', recon_loss_weight=0.0, resnet_encoder_pretrained=False, run_only_one_seed=False, sampling_batch_size=10000, save_artefacts=True, save_period=1, seed=42, skip=32, split='test', train_classifier_batch_size=128, train_classifier_classify_attributes=[], train_classifier_epochs=1, train_encoder_batch_size=500, train_encoder_classify_attributes=[], train_encoder_epochs=5, use_cuda=False, use_gen_model_reconstructions=False)


class DataManager(ABC):
    def __init__(self):
        self.dataset_cache = {}
        self.loaders_cache = {}

    @staticmethod
    def get_manager() -> 'DataManager':
        pass

    def get_dataset(self, split: str):
        assert split in ['train', 'valid', 'test']
        if split not in self.dataset_cache:
            self.dataset_cache[split] = self._get_dataset(split)
        return self.dataset_cache[split]

    @abstractmethod
    def _get_dataset(self, split: str):
        raise NotImplementedError('`get_dataset` not implemented')

    def get_dataloader(self, split: str, shuffle: bool = True, batch_size = None) -> DataLoader:
        assert split in ['train', 'valid', 'test']
        shuffle &= (split == 'train')
        if batch_size is None:
            batch_size = self.params.batch_size
        k = (split, shuffle, batch_size)
        if k not in self.loaders_cache:
            self.loaders_cache[k] = DataLoader(
                self.get_dataset(split),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0,#self.params.num_workers,
                pin_memory=True
            )
        return self.loaders_cache[k]

    @abstractmethod
    def num_classes(self) -> int:
        raise NotImplementedError('`num_classes` not implemented')

    @abstractmethod
    def target_transform(self, y):
        raise NotImplementedError('`target_transform` not implemented')

class NLIDataManager(DataManager):
    def __init__(self):
        super(NLIDataManager, self).__init__()

    def _get_dataset(self, split: str):
        raise NotImplementedError('`get_dataset` not implemented')

    def num_classes(self) -> int:
        return 3

class NLIDataset(Dataset):
    def __init__(self, split):
        dp = MNLI(root='~/Desktop/Projects/lassi/data', split=split)
        v = []
        max_prem_length, max_conc_length = 0, 0
        for _, prem, conc in dp:
            v += [[w.lower()] for w in word_tokenize(prem) + word_tokenize(conc)]
            max_prem_length = max(max_prem_length, len(prem))
            max_conc_length = max(max_conc_length, len(conc))
        self.vocab = build_vocab_from_iterator(v, specials=['<UNK>', '<PAD>'])

        UNK_IDX = self.vocab.get_stoi()['<UNK>']
        self.vocab.set_default_index(UNK_IDX)

        PAD_IDX = self.vocab.get_stoi()['<PAD>']

        self.labels = []
        self.premises = []
        self.conclusions = []

        for label, prem, conc in dp:
            tp = word_tokenize(prem)
            tc = word_tokenize(conc)
            self.labels.append(label)
            self.premises.append(self.vocab.lookup_indices(tp) + \
                                 [PAD_IDX for _ in range(max_prem_length-len(tp))])
            self.conclusions.append(self.vocab.lookup_indices(tc) + \
                                 [PAD_IDX for _ in range(max_conc_length-len(tc))])
        
        self.labels = torch.tensor(self.labels)
        self.premises = torch.tensor(self.premises)
        self.conclusions = torch.tensor(self.conclusions)
    
    def __getitem__(self, idx):
        return self.premises[idx], self.conclusions[idx], self.labels[idx]
    
    def __len__(self, idx):
        return self.labels.shape[0]

class NLIOriginalDataManager(NLIDataManager):
    def __init__(self):
        super(NLIOriginalDataManager, self).__init__()

    def _get_dataset(self, split: str):
        return NLIDataset(split)

class LSTMEncoder(nn.Module):
    def __init__(self):
        self.embedder = nn.Embedding(num_embeddings=10000, embedding_dim=768)
        self.encoder = nn.LSTM(input_size=768, hidden_size=512, batch_first=True, bidirectional=True)

    def forward(self, text_batch):
        pass

class LASSIEncoderNeg(nn.Module):
    def __init__(self):
        super(LASSIEncoder, self).__init__()
        self.encoder = None

    @staticmethod
    def latent_dimension() -> int:
        raise NotImplementedError("LASSIEncoder's latent dimension method not implemented")

    def forward(self, text_batch):
        assert self.encoder is not None
        z = self.encoder(text_batch)
        return z

class LASSIEncoderLSTM(LASSIEncoderNeg):
    def __init__(self):
        super(LASSIEncoderLSTM, self).__init__()
        self.encoder = LSTMEncoder()

    @staticmethod
    def latent_dimension() -> int:
        return 2048

class AttackNLI:

    def __init__(self, params: argparse.Namespace, gen_model_wrapper: GenModelWrapper, lassi_encoder:LASSIEncoder):
        self.gen_model_wrapper = gen_model_wrapper
        self.lassi_encoder = lassi_encoder
        self.delta = params.delta

    def compute_z_lassi(self, z_gen_model_latents_wrapped: torch.Tensor):
        enc_input = self.gen_model_wrapper.compute_encoder_input(z_gen_model_latents_wrapped)
        return self.lassi_encoder(enc_input)

    def calc_loss(self, z_gen_model_latents_wrapped: torch.Tensor, z_lassi: torch.Tensor):
        z_lassi_adv = self.compute_z_lassi(z_gen_model_latents_wrapped)
        l_2 = torch.linalg.norm(z_lassi - z_lassi_adv, ord=2, dim=1)
        return torch.clamp(l_2 - self.delta, min=0.0)



class RandomSamplingAttackNLI(AttackNLI):

    def __init__(self, params: argparse.Namespace, gen_model_wrapper: GenModelWrapper, lassi_encoder: LASSIEncoder):
        super(RandomSamplingAttack, self).__init__(params, gen_model_wrapper, lassi_encoder)

        assert 0 < len(params.perturb.split(','))
        assert params.perturb_epsilon is not None

        self.num_samples = params.random_attack_num_samples
        if len(params.perturb.split(',')) == 1:
            self.noise_adder = UniformNoiseAdder(params.perturb_epsilon)
        elif len(params.perturb.split(',')) == 2:
            self.noise_adder = UniformNoise2DAdder(params.perturb_epsilon)
        else:
            self.noise_adder = UniformNoiseNDAdder(params.perturb_epsilon)
        logger.debug(f'Use {self.noise_adder.__class__.__name__} for the RandomSamplingAttack')

    def get_adv_examples(self, z_gen_model_latents: List[torch.Tensor], z_lassi: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z_gen_model_latents_all = []
            losses_all = []
            for _ in range(self.num_samples):
                z_gen_model_latents_noisy = self.noise_adder.add_noise(
                    self.gen_model_wrapper.gen_model,
                    self.gen_model_wrapper.gen_model.wrap_latents(z_gen_model_latents)
                )
                z_gen_model_latents_all.append(z_gen_model_latents_noisy.clone().detach())
                l = self.calc_loss(z_gen_model_latents_noisy, z_lassi)
                losses_all.append(l.clone().detach())
            losses_all = torch.stack(losses_all, dim=1)
            _, idx = torch.max(losses_all, dim=1)
            adv_examples = []
            for i, sample_idx in enumerate(idx.cpu().tolist()):
                adv_examples.append(z_gen_model_latents_all[sample_idx][i])
        return torch.stack(adv_examples, 0).clone().detach().requires_grad_(True)

    def augment_data(self, z_gen_model_latents: List[torch.Tensor], y: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        assert y.dim() == 1
        y_repeated = y.repeat_interleave(self.num_samples)

        z_gen_model_latents_wrapped = self.gen_model_wrapper.gen_model.wrap_latents(z_gen_model_latents)
        assert z_gen_model_latents_wrapped.dim() == 2 and z_gen_model_latents_wrapped.size(0) == y.size(0)
        z_gen_model_latents_wrapped_repeated = z_gen_model_latents_wrapped.repeat_interleave(self.num_samples, dim=0)
        z_gen_model_latents_wrapped_repeated = self.noise_adder.add_noise(
            self.gen_model_wrapper.gen_model,
            z_gen_model_latents_wrapped_repeated
        )

        return self.compute_z_lassi(z_gen_model_latents_wrapped_repeated), y_repeated

class FairEncoderExperiment(AbstractTrainer):
    def __init__(self, encoder: LASSIEncoderNeg, data_manager: NLIDataManager):
        params = None
        super(FairEncoderExperiment, self).__init__(params, data_manager,)

        # Create models and optimizer:
        self.encoder = encoder
        self.classifier = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.Softmax())
        params_list = list(self.encoder.parameters()) + list(self.classifier.parameters())

        self.decoder = None
        self.decoder_loss_fn = None

        self.opt = optim.Adam(params_list, lr=params.lr)
        self.ce_loss_fn = nn.CrossEntropyLoss(reduction='mean')

        # Data augmentation, fairness constraints and arguments:
        self.perturb_epsilon = 0.5
        self.data_augmentation = True
        self.data_augmentor = RandomSamplingAttackNLI(params, self.gen_model_wrapper, self.encoder)
        self.adv_loss_weight = 0.1
        self.attack = RandomSamplingAttackNLI(params, self.gen_model_wrapper, self.encoder)

        # Classification and reconstruction loss weights:
        self.cls_loss_weight = 0.01

    @staticmethod
    def get_experiment_name() -> str:
        return 'snli'
        
    def _get_cls_loss_and_predictions(self, z_lassi, y):
        logits = self.classifier(z_lassi)
        classification_loss = self.ce_loss_fn(logits, y)
        predictions = logits.argmax(dim=1)
        return classification_loss, predictions

    @staticmethod
    def _get_l2_diffs_between_classes(z_lassi, y):
        assert z_lassi.dim() == 2 and y.dim() == 1 and z_lassi.size(0) == y.size(0)
        batch_size = z_lassi.size(0)

        z_lassi_1 = z_lassi.repeat(batch_size, 1)
        z_lassi_2 = z_lassi.repeat_interleave(batch_size, dim=0)

        y1 = y.repeat(batch_size)
        y2 = y.repeat_interleave(batch_size, dim=0)
        mask_eq = torch.eq(y1, y2)

        l2_diffs = torch.linalg.norm(z_lassi_1 - z_lassi_2, ord=2, dim=1)
        l2_different = torch.masked_select(l2_diffs, ~mask_eq)
        l2_same = torch.masked_select(l2_diffs, mask_eq)

        assert l2_different.shape[0] + l2_same.shape[0] == batch_size * batch_size

        return l2_different, l2_same

    def _loop(self, epoch: int, subset: str, stats_only: bool):
        # Prepare data, metrics and loggers:
        metrics = ClassificationAndFairnessMetrics(subset)
        data_loader, pbar = self.get_data_loader_and_pbar(epoch, subset)

        for x, y in data_loader:
            y = self.data_manager.target_transform(y)
            x = x.to(self.device)
            y = y.to(self.device)

            # --- Compute classification loss: ---
            z_gen_model_latents, enc_input = self.gen_model_wrapper.get_latents_and_encoder_input(x)
            z_lassi_train_mode = self.encoder(enc_input)

            if not self.data_augmentation:
                y_targets = y
                classification_loss, predictions = self._get_cls_loss_and_predictions(z_lassi_train_mode, y_targets)
            else:
                z_lassi_augmented, y_augmented = self.data_augmentor.augment_data(z_gen_model_latents, y)
                z_lassi_combined = torch.cat([z_lassi_train_mode, z_lassi_augmented])
                y_targets = torch.cat([y, y_augmented])
                classification_loss, predictions = self._get_cls_loss_and_predictions(z_lassi_combined, y_targets)

            # --- Compute class differences: ---
            l2_different, l2_same = self._get_l2_diffs_between_classes(z_lassi_train_mode, y)

            # --- Compute reconstruction loss: ---
            if self.recon_loss_weight > 0.0:
                assert self.decoder is not None
                assert self.recon_decoder_type == 'linear'
                glow_code = self.gen_model_wrapper.gen_model.wrap_latents(z_gen_model_latents)
                recon_code = self.decoder(z_lassi_train_mode)
                recon_loss = torch.linalg.norm(recon_code - glow_code, ord=2, dim=1).mean()
            else:
                recon_loss = 0.0

            # --- Compute adversarial loss: ---
            run_attack = (self.adv_loss_weight > 0.0)
            if run_attack:
                if subset == 'train' and not stats_only:
                    # Switch to eval mode for the adversarial attack.
                    self.encoder.eval()

                # Recompute z with the LASSI enc in eval mode (affects batch norm).
                z_lassi_eval_mode = self.encoder(enc_input)
                z_gen_model_latents_adv_wrapped = self.attack.get_adv_examples(z_gen_model_latents, z_lassi_eval_mode)

                enc_input_adv = self.gen_model_wrapper.compute_encoder_input(z_gen_model_latents_adv_wrapped)
                z_lassi_adv = self.encoder(enc_input_adv).detach()
                l2_diffs = torch.linalg.norm(z_lassi_eval_mode - z_lassi_adv, ord=2, dim=1)

                if subset == 'train' and not stats_only:
                    # Set back the training mode while actually computing the adversarial loss.
                    self.encoder.train()

                adv_loss = self.attack.calc_loss(z_gen_model_latents_adv_wrapped, z_lassi_train_mode).mean()
            else:
                adv_loss = 0.0

            total_loss = self.cls_loss_weight * classification_loss + \
                         self.adv_loss_weight * adv_loss + \
                         self.recon_loss_weight * recon_loss

            if subset == 'train' and not stats_only:
                self.opt.zero_grad()
                total_loss.backward()
                self.opt.step()

            # Add / compute the stats:
            metrics.add(
                num_samples_batch=x.size(0),
                total_loss_batch=total_loss.item(),
                classification_loss_batch=classification_loss.item(),
                y_targets=y_targets,
                y_pred=predictions,
                l2_different=l2_different,
                l2_same=l2_same,
                adv_loss_batch=adv_loss.item() if run_attack else 0.0,
                l2_diffs_batch=l2_diffs if run_attack else None,
                reconstruction_loss_batch=recon_loss.item() if self.recon_loss_weight > 0.0 else 0.0
            )
            pbar.set_postfix(
                tot_loss=total_loss.item(),
                cls_loss=classification_loss.item(),
                adv_loss=adv_loss.item() if run_attack else 0.0,
                l2_diff_adv=l2_diffs.mean().item() if run_attack else 0.0,
                recon_loss=recon_loss.item() if self.recon_loss_weight > 0.0 else 0.0
            )
            pbar.update()

        # Wrap up:
        pbar.close()
        metrics.report(epoch)

        if self.adv_loss_weight + self.recon_loss_weight > 0.0:
            return metrics.get_total_loss_score()
        else:
            # The lower the score, the better.
            return -metrics.get_acc_score()

    def run_loop(self, epoch: int = -1, subset: str = 'train', stats_only: bool = False):
        assert subset in ['train', 'valid', 'test']
        assert (subset == 'test' and epoch == -1) or (subset != 'test' and (epoch != -1 or stats_only))

        if subset == 'train' and not stats_only:
            self.encoder.train()
            self.classifier.train()
            if self.decoder is not None:
                self.decoder.train()
        else:
            self.encoder.eval()
            self.classifier.eval()
            if self.decoder is not None:
                self.decoder.eval()

        return self._loop(epoch, subset, stats_only)

    def train(self):
        min_valid_score = float('inf')
        best_valid_epoch = -1

        self.start_timer()
        for epoch in range(self.epochs):
            self.run_loop(epoch, 'train')
            valid_score = self.run_loop(epoch, 'valid')
            model_was_saved = self.finish_epoch(epoch)
            if model_was_saved and min_valid_score > valid_score:
                min_valid_score = valid_score
                best_valid_epoch = epoch
        self.finish_training()

        if best_valid_epoch != -1:
            logger.debug(f'Best valid epoch: {best_valid_epoch}')
            logger.debug(f'Copy lassi_encoder_{best_valid_epoch + 1}.pth --> lassi_encoder_best.pth')
            logger.debug(f'Copy classifier_{best_valid_epoch + 1}.pth --> classifier_best.pth')

            with (self.logs_dir / 'best_valid_epoch').open('w') as f:
                f.write(f'Best valid epoch: {best_valid_epoch}\n')
                f.write(f'Copy lassi_encoder_{best_valid_epoch + 1}.pth --> lassi_encoder_best.pth\n')
                f.write(f'Copy classifier_{best_valid_epoch + 1}.pth --> classifier_best.pth\n')

            shutil.copyfile(
                self.saved_models_dir / f'lassi_encoder_{best_valid_epoch + 1}.pth',
                self.saved_models_dir / f'lassi_encoder_best.pth'
            )
            shutil.copyfile(
                self.saved_models_dir / f'classifier_{best_valid_epoch + 1}.pth',
                self.saved_models_dir / f'classifier_best.pth'
            )

    def save_models(self, checkpoint: str):
        torch.save(self.encoder.state_dict(), str(self.saved_models_dir / f'lassi_encoder_{checkpoint}.pth'))
        torch.save(self.classifier.state_dict(), str(self.saved_models_dir / f'classifier_{checkpoint}.pth'))

    @staticmethod
    def load_models(params: argparse.Namespace, load_aux_classifier: bool = True, freeze: bool = True):
        # Training experiment artefacts:
        training_experiment_name = FairEncoderExperiment.get_experiment_name(params)
        models_path = utils.get_path_to('saved_models') / training_experiment_name
        if models_path.is_file():
            assert models_path.suffix == '.pth'
            models_dir = models_path.parent
            checkpoint_suffix = models_path.stem.split('_')[-1]
            load_checkpoint = -1 if checkpoint_suffix in ['last', 'best'] else int(checkpoint_suffix)
        else:
            assert models_path.is_dir(), f'Directory not found: {models_path}'
            models_dir = models_path
            load_checkpoint = -1
        assert models_dir.is_dir(), \
            'Cannot find the saved_models sub-directory corresponding to the training experiment.'

        # LASSI encoder and auxiliary classifier:
        lassi_encoder = LASSIEncoderLSTM
        lassi_encoder_checkpoint_file = utils.get_checkpoint_file(models_dir, 'lassi_encoder', load_checkpoint)
        utils.load_model(lassi_encoder, lassi_encoder_checkpoint_file, 'LASSI encoder')

        if freeze:
            lassi_encoder.requires_grad_(False)

        # Auxiliary classifier:
        if load_aux_classifier:
            aux_classifier = lassi_classifier_factory(
                params,
                input_dim=lassi_encoder.latent_dimension(),
                num_classes=3
            )
            aux_classifier_checkpoint_file = utils.get_checkpoint_file(models_dir, 'classifier', load_checkpoint)
            utils.load_model(aux_classifier, aux_classifier_checkpoint_file, 'auxiliary classifier')

            if freeze:
                aux_classifier.requires_grad_(False)

            return lassi_encoder, aux_classifier
        else:
            return lassi_encoder


