from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningModule
from torch import Tensor, nn
from torchmetrics import MeanMetric, MinMetric

from .components import RelativeError, SimpleOperatorLearningL2Loss


__all__ = ["CAMOModule"]


class CAMOModule(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.criterion = SimpleOperatorLearningL2Loss()

        self.train_re = RelativeError()
        self.val_re = RelativeError()
        self.test_re = RelativeError()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_loss_best = MinMetric()
        self.val_re_best = MinMetric()

    def forward(
        self,
        x: Tensor,
        input_pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        assert input_pos is not None
        x = torch.cat((x, input_pos), dim=-1)
        return self.net(x, input_pos, query_pos)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def model_step(
        self, batch: Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x: Tensor
        y: Tensor
        input_pos: Optional[Tensor]
        query_pos: Optional[Tensor]
        x, y, input_pos, query_pos = batch
        pred = self.forward(x, input_pos, query_pos)
        loss = self.criterion(pred, y)
        return loss, pred, y

    def training_step(
        self, batch: Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]], batch_idx: int
    ) -> Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """

        loss, pred, y = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_re(pred, y)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/relative_error", self.train_re, on_step=False, on_epoch=True, prog_bar=True)

        # check_net_value_rich(self.net)
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, pred, target = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_re(pred, target)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/relative_error", self.val_re, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        loss = self.val_loss.compute()  # get current val acc
        self.val_loss_best(loss)
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        relative_error = self.val_re.compute()
        self.val_re_best(relative_error)
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/relative_error_best", self.val_re_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, pred, target = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_re(pred, target)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/relative_error", self.test_re, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams["compile"] and stage == "fit":
            self.net = torch.compile(self.net)

    def on_before_optimizer_step(self, optimizer) -> None:
        pass

    def on_after_backward(self):
        # example to inspect gradient information in tensorboard
        # from utils import check_net_value

        # check_net_value(self.net)
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams["optimizer"](params=self.trainer.model.parameters())  # type: ignore
        if self.hparams["scheduler"] is not None:
            scheduler = self.hparams.scheduler(  # type: ignore
                optimizer=optimizer,
                total_steps=self.trainer.estimated_stepping_batches,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
