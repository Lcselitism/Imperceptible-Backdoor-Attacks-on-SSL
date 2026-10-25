import torch
import torch.nn as nn
import torch.nn.functional as F

from stegastamp_like_corruptions import (
    DifferentiableImageCorruption,
    check_image_tensor
)


# ============================================================
# stegastamp_like_model.py
#
# CIFAR-10 StegaStamp / ISSBA-style encoder-decoder model.
#
# This file contains only the model architecture:
#   1. CNN building blocks
#   2. StegaStamp-style encoder / hider
#   3. Spatial transformer
#   4. Decoder
#   5. Optional critic
#   6. Full StegaStampCIFAR wrapper
#
# Other logic is split into:
#   - stegastamp_like_corruptions.py
#   - stegastamp_like_losses.py
#   - stegastamp_like_utils.py
# ============================================================


# ============================================================
# Tensor checks
# ============================================================

def check_message_tensor(message, message_length):
    """
    Check that the message has shape:
        (B, message_length)
    """

    if message.dim() != 2:
        raise ValueError(
            f"Expected message shape (B, {message_length}), "
            f"but got {tuple(message.shape)}"
        )

    if message.size(1) != message_length:
        raise ValueError(
            f"Expected message length {message_length}, "
            f"but got {message.size(1)}"
        )


# ============================================================
# CNN building blocks
# ============================================================

class ConvBNReLU(nn.Module):
    """
    Basic convolution block:
        Conv2d -> BatchNorm2d -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    """
    Two ConvBNReLU blocks.

    This is used in the U-Net-style hider and decoder.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvBNReLU(in_channels, out_channels),
            ConvBNReLU(out_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """
    Downsampling block.

    For CIFAR-10:
        32x32 -> 16x16 -> 8x8 -> 4x4
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """
    Upsampling block with U-Net skip connection.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.conv = DoubleConv(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat([x, skip], dim=1)

        return self.conv(x)


# ============================================================
# StegaStamp-style encoder / hider
# ============================================================

class StegaStampEncoder(nn.Module):
    """
    CIFAR-10 StegaStamp-style encoder.

    The encoder hides a binary message inside an image by predicting
    a small residual.

    Flow:
        message
        -> fully connected projection
        -> small spatial message map
        -> upsample to image size
        -> concatenate with image
        -> U-Net style network
        -> residual
        -> encoded image = clean image + residual

    Args:
        message_length:
            Number of hidden bits.

        image_size:
            CIFAR-10 uses 32.

        message_map_size:
            Spatial size after message projection.
            For CIFAR-10, 8x8 is a good lightweight choice.

        message_channels:
            Number of message feature channels after projection.
            StegaStamp projects message into a spatial tensor before fusion.

        base_channels:
            Width of the CNN.

        max_residual:
            Maximum residual magnitude.
            8/255 means each pixel change is bounded to roughly 8 intensity
            levels in 8-bit image space.
    """

    def __init__(
        self,
        message_length=30,
        image_size=32,
        message_map_size=8,
        message_channels=3,
        base_channels=64,
        max_residual=8 / 255
    ):
        super().__init__()

        self.message_length = message_length
        self.image_size = image_size
        self.message_map_size = message_map_size
        self.message_channels = message_channels
        self.max_residual = max_residual

        # Project the message into a small spatial tensor.
        self.message_fc = nn.Sequential(
            nn.Linear(
                message_length,
                message_channels * message_map_size * message_map_size
            ),
            nn.ReLU(inplace=True)
        )

        # Image RGB channels + projected message channels.
        input_channels = 3 + message_channels

        # U-Net encoder side.
        self.in_block = DoubleConv(input_channels, base_channels)

        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)

        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 8)

        # U-Net decoder side.
        self.up2 = UpBlock(
            in_channels=base_channels * 8,
            skip_channels=base_channels * 4,
            out_channels=base_channels * 4
        )

        self.up1 = UpBlock(
            in_channels=base_channels * 4,
            skip_channels=base_channels * 2,
            out_channels=base_channels * 2
        )

        self.up0 = UpBlock(
            in_channels=base_channels * 2,
            skip_channels=base_channels,
            out_channels=base_channels
        )

        # Predict RGB residual.
        self.residual_head = nn.Sequential(
            nn.Conv2d(
                base_channels,
                base_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                base_channels,
                3,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )

    def build_message_map(self, message, height, width):
        """
        Project message into spatial tensor and upsample it.

        Input:
            message shape:
                (B, message_length)

        Output:
            message map shape:
                (B, message_channels, H, W)
        """

        batch_size = message.size(0)

        message_map = self.message_fc(message)

        message_map = message_map.view(
            batch_size,
            self.message_channels,
            self.message_map_size,
            self.message_map_size
        )

        message_map = F.interpolate(
            message_map,
            size=(height, width),
            mode="bilinear",
            align_corners=False
        )

        return message_map

    def forward(self, image, message):
        check_image_tensor(image)
        check_message_tensor(message, self.message_length)

        _, _, height, width = image.shape

        message_map = self.build_message_map(
            message=message,
            height=height,
            width=width
        )

        x = torch.cat([image, message_map], dim=1)

        # Down path.
        skip0 = self.in_block(x)
        skip1 = self.down1(skip0)
        skip2 = self.down2(skip1)

        x = self.down3(skip2)
        x = self.bottleneck(x)

        # Up path.
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        x = self.up0(x, skip0)

        residual = self.residual_head(x)

        # Bound residual so the trigger stays visually small.
        residual = torch.tanh(residual) * self.max_residual

        encoded_image = torch.clamp(image + residual, 0.0, 1.0)

        return encoded_image, residual


# ============================================================
# Spatial transformer for decoder
# ============================================================

class SpatialTransformer(nn.Module):
    """
    Lightweight spatial transformer.

    Original StegaStamp uses a spatial transformer in the decoder to improve
    robustness against perspective/alignment changes.

    This CIFAR-10 version predicts a small affine transformation and is
    initialized as identity, so it does not disturb the image at the start.
    """

    def __init__(self):
        super().__init__()

        self.localization = nn.Sequential(
            ConvBNReLU(3, 16),
            nn.MaxPool2d(2),

            ConvBNReLU(16, 32),
            nn.MaxPool2d(2),

            ConvBNReLU(32, 64),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6)
        )

        self.init_identity()

    def init_identity(self):
        """
        Initialize affine transform as identity.
        """

        nn.init.zeros_(self.fc[-1].weight)

        identity_bias = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            dtype=torch.float32
        )

        with torch.no_grad():
            self.fc[-1].bias.copy_(identity_bias)

    def forward(self, image):
        check_image_tensor(image)

        batch_size = image.size(0)

        features = self.localization(image)
        features = torch.flatten(features, start_dim=1)

        theta = self.fc(features)
        theta = theta.view(batch_size, 2, 3)

        grid = F.affine_grid(
            theta,
            image.size(),
            align_corners=False
        )

        transformed = F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=False
        )

        return transformed


# ============================================================
# Decoder
# ============================================================

class StegaStampDecoder(nn.Module):
    """
    Decoder network.

    The decoder receives the encoded or corrupted image and predicts the
    hidden message bits.

    Output:
        raw logits of shape (B, message_length)

    During training:
        use BCEWithLogitsLoss.
    """

    def __init__(
        self,
        message_length=30,
        base_channels=64,
        use_spatial_transformer=True
    ):
        super().__init__()

        self.message_length = message_length
        self.use_spatial_transformer = use_spatial_transformer

        if use_spatial_transformer:
            self.stn = SpatialTransformer()
        else:
            self.stn = nn.Identity()

        self.features = nn.Sequential(
            DoubleConv(3, base_channels),
            nn.MaxPool2d(2),          # 32 -> 16

            DoubleConv(base_channels, base_channels * 2),
            nn.MaxPool2d(2),          # 16 -> 8

            DoubleConv(base_channels * 2, base_channels * 4),
            nn.MaxPool2d(2),          # 8 -> 4

            DoubleConv(base_channels * 4, base_channels * 4),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, message_length)
        )

    def forward(self, image):
        check_image_tensor(image)

        x = self.stn(image)

        x = self.features(x)
        x = torch.flatten(x, start_dim=1)

        logits = self.classifier(x)

        return logits


# ============================================================
# Optional critic
# ============================================================

class StegaStampCritic(nn.Module):
    """
    Optional critic network.

    Original StegaStamp uses a critic loss to improve perceptual quality.
    For now, this is included as a future extension but does not need to
    be used in the first stable experiment.

    Output:
        one score per image.
    """

    def __init__(self, base_channels=64):
        super().__init__()

        self.features = nn.Sequential(
            ConvBNReLU(3, base_channels),
            nn.MaxPool2d(2),

            ConvBNReLU(base_channels, base_channels * 2),
            nn.MaxPool2d(2),

            ConvBNReLU(base_channels * 2, base_channels * 4),
            nn.MaxPool2d(2),

            ConvBNReLU(base_channels * 4, base_channels * 4),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.head = nn.Linear(base_channels * 4, 1)

    def forward(self, image):
        check_image_tensor(image)

        x = self.features(image)
        x = torch.flatten(x, start_dim=1)

        score = self.head(x).squeeze(1)

        return score


# ============================================================
# Full model wrapper
# ============================================================

class StegaStampCIFAR(nn.Module):
    """
    Full CIFAR-10 StegaStamp / ISSBA-style model.

    Training flow:
        clean image + message
        -> encoder
        -> encoded image
        -> differentiable corruption layer
        -> decoder
        -> decoded message logits

    Inference / trigger generation flow:
        clean image + fixed target message
        -> encoder
        -> encoded image

    The encoded image is the poisoned image.
    The residual is the invisible sample-specific trigger.
    """

    def __init__(
        self,
        message_length=30,
        image_size=32,
        message_map_size=8,
        message_channels=3,
        base_channels=64,
        max_residual=8 / 255,
        use_spatial_transformer=True,
        use_corruption=True
    ):
        super().__init__()

        self.message_length = message_length
        self.image_size = image_size
        self.max_residual = max_residual
        self.use_corruption = use_corruption

        self.encoder = StegaStampEncoder(
            message_length=message_length,
            image_size=image_size,
            message_map_size=message_map_size,
            message_channels=message_channels,
            base_channels=base_channels,
            max_residual=max_residual
        )

        self.decoder = StegaStampDecoder(
            message_length=message_length,
            base_channels=base_channels,
            use_spatial_transformer=use_spatial_transformer
        )

        if use_corruption:
            self.corruption = DifferentiableImageCorruption(
                image_size=image_size
            )
        else:
            self.corruption = None

    def forward(
        self,
        image,
        message,
        apply_corruption=True,
        corruption_strength=1.0
    ):
        """
        Forward pass for encoder-decoder training.

        Returns a dictionary because training needs several intermediate
        tensors for losses and logging.
        """

        check_image_tensor(image)
        check_message_tensor(message, self.message_length)

        encoded_image, residual = self.encoder(image, message)

        if (
            apply_corruption
            and self.use_corruption
            and self.corruption is not None
        ):
            corrupted_image = self.corruption(
                encoded_image,
                strength=corruption_strength
            )
        else:
            corrupted_image = encoded_image

        decoded_logits = self.decoder(corrupted_image)

        return {
            "encoded_image": encoded_image,
            "residual": residual,
            "corrupted_image": corrupted_image,
            "decoded_logits": decoded_logits
        }

    @torch.no_grad()
    def encode(self, image, message):
        """
        Encode a message into an image.

        Use this during SL and SSL backdoor experiments.
        """

        self.eval()

        check_image_tensor(image)
        check_message_tensor(message, self.message_length)

        encoded_image, residual = self.encoder(image, message)

        return encoded_image, residual

    @torch.no_grad()
    def decode(self, image):
        """
        Decode hidden message logits from an encoded image.
        """

        self.eval()

        check_image_tensor(image)

        logits = self.decoder(image)

        return logits


# ============================================================
# Backward-compatible alias
# ============================================================

# Some older files may still import StegaStampLike.
# Later, we will update those files properly, but this alias helps avoid
# immediate import errors while transitioning.
StegaStampLike = StegaStampCIFAR


# ============================================================
# Quick sanity check
# ============================================================

if __name__ == "__main__":
    from stegastamp_like_utils import (
        create_random_message,
        bit_accuracy,
        compute_psnr,
        compute_residual_stats
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StegaStampCIFAR(
        message_length=30,
        image_size=32,
        message_map_size=8,
        message_channels=3,
        base_channels=32,
        max_residual=8 / 255,
        use_spatial_transformer=True,
        use_corruption=True
    ).to(device)

    images = torch.rand(4, 3, 32, 32).to(device)

    messages = create_random_message(
        batch_size=4,
        message_length=30,
        device=device
    )

    model.train()

    outputs = model(
        image=images,
        message=messages,
        apply_corruption=True,
        corruption_strength=0.5
    )

    encoded_images = outputs["encoded_image"]
    residuals = outputs["residual"]
    corrupted_images = outputs["corrupted_image"]
    decoded_logits = outputs["decoded_logits"]

    acc = bit_accuracy(decoded_logits, messages)
    psnr = compute_psnr(encoded_images, images)
    residual_stats = compute_residual_stats(residuals)

    print("Input images:", images.shape)
    print("Encoded images:", encoded_images.shape)
    print("Residuals:", residuals.shape)
    print("Corrupted images:", corrupted_images.shape)
    print("Decoded logits:", decoded_logits.shape)
    print("Initial bit accuracy:", acc)
    print("Initial PSNR:", psnr)
    print("Residual stats:", residual_stats)