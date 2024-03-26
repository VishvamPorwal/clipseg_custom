import itertools
from pytorch_lightning import LightningModule
import torch
from torch import nn
from transformers import AutoImageProcessor, AutoModel
import transformers
from torch.functional import F
from torch import optim
from torchvision import transforms



class ImageEncoderDinov2(nn.Module):
    def __init__(self, model_name, trainable=True):
        super().__init__()
        self.processor = transforms.Compose([
                            transforms.Normalize(
                                mean=(123.675, 116.28, 103.53),
                                std=(58.395, 57.12, 57.375),
                            ),
                            # transforms.Resize(56),
                        ])
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        for param in self.encoder.parameters():
            param.requires_grad = trainable
        
        self.target_token_idx = 0

    def forward(self, images):
        inputs = self.processor(images)
        inputs = inputs.to('cuda')
        # inputs['pixel_values'] = inputs['pixel_values'].to('cuda')
        outputs = self.encoder(inputs)
        return outputs

class ImageEncoderhf(nn.Module):
    def __init__(self, model_name, trainable=True):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, output_hidden_states=True)

        for param in self.encoder.parameters():
            param.requires_grad = trainable
        
        self.target_token_idx = 0

    def forward(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].to('cuda')
        outputs = self.encoder(**inputs)
        return outputs.last_hidden_state[:, self.target_token_idx, :]

class TextEncoder(nn.Module):
    def __init__(self, model_name: str, trainable: bool = True) -> None:
        super().__init__()


        self.model = transformers.AutoModel.from_pretrained(model_name)


        for param in self.model.parameters():
            param.requires_grad = trainable


        self.target_token_idx = 0


    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]
    

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()


        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)


        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)


    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)


        x += projected


        return self.layer_norm(x)


class ClipModel(LightningModule):
    def __init__(
        self,
        image_encoder_alias: str,
        text_encoder_alias: str,
        # image_encoder_pretrained: bool = True,
        image_encoder_trainable: bool = True,
        text_encoder_trainable: bool = True,
        image_embedding_dims: int = 2048,
        text_embedding_dims: int = 768,
        projection_dims: int = 256,
        dropout: float = 0.0,
        temperature: float = 1.0,
        weight_decay: float = 0.0,
        head_lr: float = 1e-3,
        image_encoder_lr: float = 1e-4,
        text_encoder_lr: float = 1e-5,
        lr_scheduler_patience: float = 1.0,
        lr_scheduler_factor: float = 0.8,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if image_encoder_alias == "facebook/dinov2-small":
            self.image_encoder = ImageEncoderDinov2(
                model_name=image_encoder_alias,
                trainable=image_encoder_trainable,
            )
        else:
            self.image_encoder = ImageEncoderhf(
                model_name=image_encoder_alias,
                trainable=image_encoder_trainable,
            )
        self.text_encoder = TextEncoder(
            model_name=text_encoder_alias, trainable=text_encoder_trainable
        )
        self.image_projection = ProjectionHead(
            embedding_dim=image_embedding_dims,
            projection_dim=projection_dims,
            dropout=dropout,
        )
        self.text_projection = ProjectionHead(
            embedding_dim=text_embedding_dims,
            projection_dim=projection_dims,
            dropout=dropout,
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.head_lr = head_lr
        self.image_encoder_lr = image_encoder_lr
        self.text_encoder_lr = text_encoder_lr
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor


        self.save_hyperparameters()


    def _compute_losses(self, image_embeddings, text_embeddings):
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        images_loss = (-targets.T * self.log_softmax(logits.T)).sum(1)
        texts_loss = (-targets * self.log_softmax(logits)).sum(1)
        return (images_loss + texts_loss) / 2.0

    def forward(self, inputs):
        image_features = self.image_encoder(inputs["image"])
        text_features = self.text_encoder(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )


        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)


        return image_embeddings, text_embeddings


    def configure_optimizers(self):
        parameters = [
            {"params": self.image_encoder.parameters(), "lr": self.image_encoder_lr},
            {"params": self.text_encoder.parameters(), "lr": self.text_encoder_lr},
            {
                "params": itertools.chain(
                    self.image_projection.parameters(),
                    self.text_projection.parameters(),
                ),
                "lr": self.head_lr,
                "weight_decay": self.weight_decay,
            },
        ]
        optimizer = optim.Adam(parameters, weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.lr_scheduler_patience,
            factor=self.lr_scheduler_factor,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val/loss",
        }


    def training_step(self, batch, *args, **kwargs):
        image_embeddings, text_embeddings = self.forward(batch)
        loss = self._compute_losses(image_embeddings, text_embeddings).mean()
        train_loss = self.all_gather(loss)
        self.log("train/loss", train_loss.mean())
        return loss

    def validation_step(self, batch, *args, **kwargs):
        image_embeddings, text_embeddings = self.forward(batch)
        loss = self._compute_losses(image_embeddings, text_embeddings).mean()
        val_loss = self.all_gather(loss)
        self.log("val/loss", val_loss.mean())
        # accuracy = self.calculate_accuracy(batch)
        # self.log("val/accuracy", accuracy)
        return loss
    
    # def calculate_accuracy(self, batch):
    #     image_embeddings, text_embeddings = self.forward(batch)
    #     logits = (text_embeddings @ image_embeddings.T) / self.temperature
    #     targets = torch.arange(len(logits)).to(logits.device)
    #     return (logits.argmax(dim=-1) == targets).float().mean()

        
    def predict(self, image, texts):
        # find nearest text to image
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(texts)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        return logits.argmax()
