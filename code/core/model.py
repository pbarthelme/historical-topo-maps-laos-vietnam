import lightning as L
import segmentation_models_pytorch as smp
import torch
import torchmetrics
import torch.nn as nn


class SegmentationModelLightning(L.LightningModule):
    """
    A PyTorch Lightning module for semantic segmentation.

    This module supports various segmentation architectures (e.g., UNet, DeepLabV3) based on the 
    segmentation-models-pytorch package and includes training, validation, and prediction steps,
    along with metrics for evaluation.

    Parameters:
        model_name (str): Name of the model architecture (e.g., 'unet', 'unetplusplus').
        encoder_name (str): Name of the encoder to use (e.g., 'resnet50').
        encoder_weights (str): Pre-trained weights for the encoder (e.g., 'imagenet').
        in_channels (int): Number of input channels (e.g., 3 for RGB images).
        classes (int): Number of output classes.
        activation (str, optional): Activation function to apply to the output (e.g., 'softmax', 'sigmoid').
        lr (float): Learning rate for the optimizer.
        loss_fn (nn.Module): Loss function to use during training (e.g., nn.CrossEntropyLoss).
    """
    def __init__(
        self,
        model_name: str = 'unet',
        encoder_name: str = 'resnet50',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 1,
        activation: str = None,
        lr: float = 1e-3,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        ignore_index: int = None,
    ):
        """Initializes the segmentation model with the specified architecture."""
        super().__init__()
        self.save_hyperparameters()

        # Initialize the chosen model architecture
        self.model = self._initialize_model(
            model_name=model_name,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )

        self.loss_fn = loss_fn
        self.lr = lr

        # Metrics
        self.valid_macro_f1 = torchmetrics.F1Score(task="multiclass", num_classes=classes, average="macro", ignore_index=ignore_index)
        self.valid_micro_f1 = torchmetrics.F1Score(task="multiclass", num_classes=classes, average="micro", ignore_index=ignore_index)
        self.valid_macro_precision = torchmetrics.Precision(task="multiclass", num_classes=classes, average="macro", ignore_index=ignore_index)
        self.valid_micro_precision = torchmetrics.Precision(task="multiclass", num_classes=classes, average="micro", ignore_index=ignore_index)
        self.valid_macro_recall = torchmetrics.Recall(task="multiclass", num_classes=classes, average="macro", ignore_index=ignore_index)
        self.valid_micro_recall = torchmetrics.Recall(task="multiclass", num_classes=classes, average="micro", ignore_index=ignore_index)
        self.valid_macro_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=classes, average="macro", ignore_index=ignore_index)
        self.valid_micro_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=classes, average="micro", ignore_index=ignore_index)

    def _initialize_model(self, **kwargs) -> nn.Module:
        """
        Initializes the segmentation-models-pytorch model based on the specified architecture.

        Parameters:
            **kwargs: Parameters for model initialization.

        Returns:
            nn.Module: The initialized model.
        """
        model_name = kwargs.pop("model_name").lower()
        if model_name == "unet":
            return smp.Unet(**kwargs)
        elif model_name in ["unet++", "unetplusplus"]:
            return smp.UnetPlusPlus(**kwargs)
        elif model_name in ["deeplabv3+", "deeplabv3plus"]:
            return smp.DeepLabV3Plus(**kwargs)
        elif model_name == "segformer":
            return smp.Segformer(**kwargs)
        else:
            raise ValueError(f"Model '{model_name}' is not supported.")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        images, masks = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, masks)
        self.log("train_loss", loss, prog_bar=False, on_step=False, on_epoch=True, logger=True)       
        return loss

    def validation_step(self, batch):
        images, masks = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, masks)

        self.valid_macro_f1(outputs, masks)
        self.valid_micro_f1(outputs, masks)
        self.valid_macro_precision(outputs, masks)
        self.valid_micro_precision(outputs, masks)
        self.valid_macro_recall(outputs, masks)
        self.valid_micro_recall(outputs, masks)
        self.valid_macro_accuracy(outputs, masks)
        self.valid_micro_accuracy(outputs, masks)

        self.log('val_loss', loss, prog_bar=False, on_step=False, on_epoch=True, logger=True)
        self.log('valid_macro_f1', self.valid_macro_f1, prog_bar=False, on_step=False, on_epoch=True, logger=True)
        self.log('valid_micro_f1', self.valid_micro_f1, prog_bar=False, on_step=False, on_epoch=True, logger=True)
        self.log('valid_macro_precision', self.valid_macro_precision, prog_bar=False, on_step=False, on_epoch=True, logger=True)
        self.log('valid_micro_precision', self.valid_micro_precision, prog_bar=False, on_step=False, on_epoch=True, logger=True)
        self.log('valid_macro_recall', self.valid_macro_recall, prog_bar=False, on_step=False, on_epoch=True, logger=True)
        self.log('valid_micro_recall', self.valid_micro_recall, prog_bar=False, on_step=False, on_epoch=True, logger=True)
        self.log('valid_macro_accuracy', self.valid_macro_accuracy, prog_bar=False, on_step=False, on_epoch=True, logger=True)
        self.log('valid_micro_accuracy', self.valid_micro_accuracy, prog_bar=False, on_step=False, on_epoch=True, logger=True)
        return outputs

    def predict_step(self, batch):
        outputs = self.model(batch)
        outputs_class = torch.argmax(outputs, dim=1).cpu().numpy()

        return outputs_class
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
