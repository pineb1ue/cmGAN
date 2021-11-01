from tqdm import tqdm
from os.path import join
from typing import Any

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter


class _Trainer:
    """
    Class that represents the whole training
    """

    def __init__(
        self,
        train_loader: torch.utils.data.DataLoader,
        feature_generator: torchvision.models,
        id_classifier: torchvision.models,
        domain_classifier: torchvision.models,
        criterion_triplet: torch.nn.functional,
        criterion_id: torch.nn.functional,
        criterion_domain: torch.nn.functional,
        optimizer: torch.optim,
        lr_scheduler: torch.optim,
        num_epochs: int,
        device: str,
        interval: int,
        output_dir: str,
        project_name: str,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.05,
        K: int = 5,
        nu: int = 1,
    ):
        self._train_loader = train_loader
        self._feature_generator = feature_generator
        self._id_classifier = id_classifier
        self._domain_classifier = domain_classifier
        self._criterion_triplet = criterion_triplet
        self._criterion_id = criterion_id
        self._criterion_domain = criterion_domain
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._num_epochs = num_epochs
        self._device = device
        self._interval = interval
        self._output_dir = output_dir

        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._K = K
        self._nu = nu

        self._feature_generator = self._feature_generator.to(self._device)
        self._id_classifier = self._id_classifier.to(self._device)
        self._domain_classifier = self._domain_classifier.to(self._device)
        self._writer = SummaryWriter(project_name)

    def fit(self):
        for epoch in tqdm(range(self._num_epochs)):
            self._train_epoch(epoch)

            self._writer.add_scalar("lr", self._get_lr(self._optimizer), epoch + 1)

            if (epoch + 1) % self._interval == 0:
                self._save_model(epoch)

    def _train_epoch(self, epoch: int):
        pass

    def _save_model(self, keyword: Any):
        if isinstance(keyword, (int, float)):
            filename = "triplet_{:>04d}.pth".format(keyword + 1)
            self._save_path = join(self._output_dir, filename)
            torch.save(self._feature_generator.state_dict(), self._save_path)
        else:
            filename = "triplet_{}.pth".format(keyword)
            self._save_path = join(self._output_dir, filename)
            torch.save(self._best_model.state_dict(), self._save_path)

    @staticmethod
    def _get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]


class TripletTrainer(_Trainer):
    """
    Class that represents the triplet training
    """

    def _train_epoch(self, epoch: int):
        self._feature_generator.train()
        self._id_classifier.train()
        self._domain_classifier.train()
        epoch_loss = 0.0
        epoch_triplet_loss = 0.0
        epoch_id_loss = 0.0
        epoch_domain_loss = 0.0
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0

        for batch in self._train_loader:

            (
                anc_img_3d,
                pos_img_3d,
                neg_img_3d,
                anc_img_2d,
                pos_img_2d,
                neg_img_2d,
                anc_label,
                domain_3d,
                domain_2d,
            ) = batch

            # GPU
            anc_img_3d = anc_img_3d.to(self._device)
            pos_img_3d = pos_img_3d.to(self._device)
            neg_img_3d = neg_img_3d.to(self._device)
            anc_img_2d = anc_img_2d.to(self._device)
            pos_img_2d = pos_img_2d.to(self._device)
            neg_img_2d = neg_img_2d.to(self._device)
            anc_label = anc_label.to(self._device)
            domain_3d = domain_3d.to(self._device)
            domain_2d = domain_2d.to(self._device)

            self._optimizer.zero_grad()

            # ------------------------------
            # Generator
            # ------------------------------

            anc_3d_features = self._feature_generator(anc_img_3d)
            pos_3d_features = self._feature_generator(pos_img_3d)
            neg_3d_features = self._feature_generator(neg_img_3d)

            anc_2d_features = self._feature_generator(anc_img_2d)
            pos_2d_features = self._feature_generator(pos_img_2d)
            neg_2d_features = self._feature_generator(neg_img_2d)

            # Triplet Loss
            triplet_loss_3d = self._criterion_triplet(
                anc_3d_features, pos_2d_features, neg_2d_features
            )
            triplet_loss_2d = self._criterion_triplet(
                anc_2d_features, pos_3d_features, neg_3d_features
            )
            triplet_loss = triplet_loss_3d + triplet_loss_2d

            # Class Loss
            id_loss_3d = self._criterion_id(
                self._id_classifier(anc_3d_features), anc_label
            )
            id_loss_2d = self._criterion_id(
                self._id_classifier(anc_2d_features), anc_label
            )
            id_loss = id_loss_3d + id_loss_2d

            # Generator Loss
            loss_G = self._alpha * triplet_loss + self._beta * id_loss

            # ------------------------------
            # Discriminator
            # ------------------------------

            precited_domain_3d = self._domain_classifier(anc_3d_features)
            precited_domain_2d = self._domain_classifier(anc_2d_features)
            domain_loss_3d = self._criterion_domain(precited_domain_3d, domain_3d)
            domain_loss_2d = self._criterion_domain(precited_domain_2d, domain_2d)
            domain_loss = domain_loss_3d + domain_loss_2d
            loss_D = domain_loss

            if epoch % self._K:
                loss_total = loss_G - self._gamma * loss_D
            else:
                loss_total = self._nu * (self._gamma * loss_D - loss_G)

            epoch_loss += loss_total.item()
            epoch_triplet_loss += triplet_loss.item()
            epoch_id_loss += id_loss.item()
            epoch_domain_loss += domain_loss.item()
            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()

            loss_total.backward()
            self._optimizer.step()

        self._lr_scheduler.step()

        print(
            "Train : [Epoch %d/%d] [loss: %f]"
            % (
                epoch,
                self._num_epochs,
                epoch_loss,
            )
        )

        self._writer.add_scalar(
            "train/loss",
            epoch_loss,
            epoch + 1,
        )
        self._writer.add_scalar(
            "train/triplet_loss",
            epoch_triplet_loss,
            epoch + 1,
        )
        self._writer.add_scalar(
            "train/id_loss",
            epoch_id_loss,
            epoch + 1,
        )
        self._writer.add_scalar(
            "train/domain_loss",
            epoch_domain_loss,
            epoch + 1,
        )
        self._writer.add_scalar(
            "train/loss_G",
            epoch_loss_G,
            epoch + 1,
        )
        self._writer.add_scalar(
            "train/loss_D",
            epoch_loss_D,
            epoch + 1,
        )
