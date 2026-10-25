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
# Advanced CIFAR-10 StegaStamp / ISSBA-style encoder-decoder.
#
# CA3 version:
#   1. Advanced multi-scale steganographic hider
#   2. Message FiLM conditioning at multiple resolutions
#   3. GroupNorm-based residual blocks for the hider
#   4. Adaptive residual mask
#   5. Strict bounded residual
#   6. Existing decoder / corruption / wrapper compatibility
#
# Other logic remains split into:
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
# General helper blocks
# ============================================================

def get_group_count(channels, preferred_groups=8):
    """
    Find a valid number of groups for GroupNorm.

    GroupNorm requires:
        channels % num_groups == 0
    """

    if channels <= 0:
        raise ValueError("channels must be greater than 0")

    for group_count in range(min(preferred_groups, channels), 0, -1):
        if channels % group_count == 0:
            return group_count

    return 1


class ConvBNReLU(nn.Module):
    """
    Basic convolution block:
        Conv2d -> BatchNorm2d -> ReLU

    This is kept for the decoder, spatial transformer, and critic to remain
    close to your CA2 baseline.
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

    Used by decoder / spatial transformer / critic.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvBNReLU(in_channels, out_channels),
            ConvBNReLU(out_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)


class ConvGNAct(nn.Module):
    """
    Conv2d -> GroupNorm -> SiLU.

    This is used in the advanced hider. GroupNorm avoids making the trigger
    generated for one image depend on other images in the same batch.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        activation=True
    ):
        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels must be greater than 0")

        if out_channels <= 0:
            raise ValueError("out_channels must be greater than 0")

        padding = kernel_size // 2

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.GroupNorm(
                num_groups=get_group_count(out_channels),
                num_channels=out_channels
            )
        ]

        if activation:
            layers.append(nn.SiLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualConvBlock(nn.Module):
    """
    Residual convolution block for the advanced hider.

    Structure:
        ConvGNAct -> ConvGN -> skip -> SiLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = ConvGNAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            activation=True
        )

        self.conv2 = ConvGNAct(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            activation=False
        )

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )

        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = x + identity
        x = self.activation(x)

        return x


# ============================================================
# Advanced hider message conditioning blocks
# ============================================================

class SpatialMessageMap(nn.Module):
    """
    Project the binary message into a small spatial tensor and upsample it
    to the image size.

    This keeps the original StegaStamp idea:
        message -> spatial map -> concatenate with image
    """

    def __init__(
        self,
        message_length,
        message_channels,
        message_map_size
    ):
        super().__init__()

        self.message_length = message_length
        self.message_channels = message_channels
        self.message_map_size = message_map_size

        flattened_size = (
            message_channels
            * message_map_size
            * message_map_size
        )

        self.projection = nn.Sequential(
            nn.Linear(message_length, flattened_size),
            nn.SiLU(inplace=True)
        )

    def forward(self, message, output_size):
        batch_size = message.size(0)

        message_map = self.projection(message)

        message_map = message_map.view(
            batch_size,
            self.message_channels,
            self.message_map_size,
            self.message_map_size
        )

        message_map = F.interpolate(
            message_map,
            size=output_size,
            mode="bilinear",
            align_corners=False
        )

        return message_map


class MessageFiLM(nn.Module):
    """
    Message-conditioned Feature-wise Linear Modulation.

    For each feature channel, the hidden message predicts:
        gamma: channel scale
        beta: channel bias

    Then:
        output = features * (1 + gamma) + beta

    This injects the message repeatedly at different U-Net resolutions.
    """

    def __init__(
        self,
        message_length,
        feature_channels,
        hidden_channels=128,
        modulation_strength=0.10
    ):
        super().__init__()

        self.feature_channels = feature_channels
        self.modulation_strength = float(modulation_strength)

        self.conditioner = nn.Sequential(
            nn.Linear(message_length, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, feature_channels * 2)
        )

        # Identity-like start:
        # gamma = 0, beta = 0 at initialization.
        final_layer = self.conditioner[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(self, features, message):
        parameters = self.conditioner(message)

        gamma, beta = parameters.chunk(2, dim=1)

        gamma = gamma.view(
            -1,
            self.feature_channels,
            1,
            1
        )

        beta = beta.view(
            -1,
            self.feature_channels,
            1,
            1
        )

        gamma = self.modulation_strength * torch.tanh(gamma)
        beta = self.modulation_strength * torch.tanh(beta)

        return features * (1.0 + gamma) + beta


class AdvancedDownBlock(nn.Module):
    """
    Downsample by 2, refine with residual block, then inject message.
    """

    def __init__(self, in_channels, out_channels, message_length):
        super().__init__()

        self.downsample = ConvGNAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            activation=True
        )

        self.residual = ResidualConvBlock(
            in_channels=out_channels,
            out_channels=out_channels
        )

        self.message_film = MessageFiLM(
            message_length=message_length,
            feature_channels=out_channels
        )

    def forward(self, x, message):
        x = self.downsample(x)
        x = self.residual(x)
        x = self.message_film(x, message)

        return x


class AdvancedUpBlock(nn.Module):
    """
    Upsample, concatenate skip connection, refine, then inject message.
    """

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        message_length
    ):
        super().__init__()

        self.fusion = ResidualConvBlock(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels
        )

        self.message_film = MessageFiLM(
            message_length=message_length,
            feature_channels=out_channels
        )

    def forward(self, x, skip, message):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat([x, skip], dim=1)

        x = self.fusion(x)
        x = self.message_film(x, message)

        return x


# ============================================================
# Advanced StegaStamp-style encoder / hider
# ============================================================

class StegaStampEncoder(nn.Module):
    """
    Advanced CIFAR-10 StegaStamp-style encoder / hider.

    Input:
        clean image x in [0, 1]
        binary message m

    Output:
        encoded image x'
        effective residual x' - x

    Main CA3 upgrades:
        1. Multi-scale message injection
        2. Message FiLM conditioning
        3. GroupNorm residual blocks
        4. Adaptive residual mask
        5. Strict residual bound
    """

    def __init__(
        self,
        message_length=16,
        image_size=32,
        message_map_size=8,
        message_channels=8,
        base_channels=32,
        max_residual=8 / 255
    ):
        super().__init__()

        if message_length <= 0:
            raise ValueError("message_length must be greater than 0")

        if image_size <= 0:
            raise ValueError("image_size must be greater than 0")

        if message_map_size <= 0:
            raise ValueError("message_map_size must be greater than 0")

        if message_channels <= 0:
            raise ValueError("message_channels must be greater than 0")

        if base_channels <= 0:
            raise ValueError("base_channels must be greater than 0")

        if max_residual <= 0.0:
            raise ValueError("max_residual must be greater than 0")

        self.message_length = message_length
        self.image_size = image_size
        self.message_map_size = message_map_size
        self.message_channels = message_channels
        self.base_channels = base_channels
        self.max_residual = float(max_residual)

        self.message_map = SpatialMessageMap(
            message_length=message_length,
            message_channels=message_channels,
            message_map_size=message_map_size
        )

        input_channels = 3 + message_channels

        # 32 x 32
        self.input_block = ResidualConvBlock(
            in_channels=input_channels,
            out_channels=base_channels
        )

        self.input_message_film = MessageFiLM(
            message_length=message_length,
            feature_channels=base_channels
        )

        # 32 -> 16
        self.down1 = AdvancedDownBlock(
            in_channels=base_channels,
            out_channels=base_channels * 2,
            message_length=message_length
        )

        # 16 -> 8
        self.down2 = AdvancedDownBlock(
            in_channels=base_channels * 2,
            out_channels=base_channels * 4,
            message_length=message_length
        )

        # 8 -> 4
        self.down3 = AdvancedDownBlock(
            in_channels=base_channels * 4,
            out_channels=base_channels * 8,
            message_length=message_length
        )

        self.bottleneck = nn.Sequential(
            ResidualConvBlock(
                in_channels=base_channels * 8,
                out_channels=base_channels * 8
            ),
            ResidualConvBlock(
                in_channels=base_channels * 8,
                out_channels=base_channels * 8
            )
        )

        self.bottleneck_message_film = MessageFiLM(
            message_length=message_length,
            feature_channels=base_channels * 8
        )

        # 4 -> 8
        self.up2 = AdvancedUpBlock(
            in_channels=base_channels * 8,
            skip_channels=base_channels * 4,
            out_channels=base_channels * 4,
            message_length=message_length
        )

        # 8 -> 16
        self.up1 = AdvancedUpBlock(
            in_channels=base_channels * 4,
            skip_channels=base_channels * 2,
            out_channels=base_channels * 2,
            message_length=message_length
        )

        # 16 -> 32
        self.up0 = AdvancedUpBlock(
            in_channels=base_channels * 2,
            skip_channels=base_channels,
            out_channels=base_channels,
            message_length=message_length
        )

        self.residual_content_head = nn.Sequential(
            ResidualConvBlock(
                in_channels=base_channels,
                out_channels=base_channels
            ),
            nn.Conv2d(
                base_channels,
                3,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )

        mask_hidden_channels = max(base_channels // 2, 8)

        self.residual_mask_head = nn.Sequential(
            ConvGNAct(
                in_channels=base_channels,
                out_channels=mask_hidden_channels,
                kernel_size=3,
                stride=1,
                activation=True
            ),
            nn.Conv2d(
                mask_hidden_channels,
                1,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )

        self.initialize_output_heads()

    def initialize_output_heads(self):
        """
        Start with a small residual.

        The final residual is:
            tanh(content) * sigmoid(mask) * max_residual

        Initial mask bias -1.5 gives sigmoid(-1.5) ≈ 0.18.
        """

        residual_layer = self.residual_content_head[-1]

        nn.init.normal_(
            residual_layer.weight,
            mean=0.0,
            std=1e-3
        )
        nn.init.zeros_(residual_layer.bias)

        mask_layer = self.residual_mask_head[-1]

        nn.init.zeros_(mask_layer.weight)
        nn.init.constant_(mask_layer.bias, -1.5)

    def generate_trigger_with_mask(self, image, message):
        """
        Internal hider implementation.

        Returns:
            encoded_image:
                final triggered image in [0, 1]

            effective_residual:
                encoded_image - image

            residual_mask:
                learned spatial mask in [0, 1]
        """

        check_image_tensor(image)
        check_message_tensor(message, self.message_length)

        if image.size(0) != message.size(0):
            raise ValueError(
                "Image batch size and message batch size must match. "
                f"Got image batch {image.size(0)} and "
                f"message batch {message.size(0)}."
            )

        _, _, height, width = image.shape

        message_map = self.message_map(
            message=message,
            output_size=(height, width)
        )

        x = torch.cat([image, message_map], dim=1)

        # 32 x 32
        skip0 = self.input_block(x)
        skip0 = self.input_message_film(skip0, message)

        # 16 x 16
        skip1 = self.down1(skip0, message)

        # 8 x 8
        skip2 = self.down2(skip1, message)

        # 4 x 4
        x = self.down3(skip2, message)

        x = self.bottleneck(x)
        x = self.bottleneck_message_film(x, message)

        # 8 x 8
        x = self.up2(x, skip2, message)

        # 16 x 16
        x = self.up1(x, skip1, message)

        # 32 x 32
        x = self.up0(x, skip0, message)

        residual_content = self.residual_content_head(x)
        residual_content = torch.tanh(residual_content)

        residual_mask = self.residual_mask_head(x)
        residual_mask = torch.sigmoid(residual_mask)

        bounded_residual = (
            residual_content
            * residual_mask
            * self.max_residual
        )

        encoded_image = torch.clamp(
            image + bounded_residual,
            min=0.0,
            max=1.0
        )

        # This is the true final residual after clamping.
        effective_residual = encoded_image - image

        return encoded_image, effective_residual, residual_mask

    def forward(self, image, message):
        """
        Standard forward method.

        Compatible with existing CA2 scripts:
            encoded_image, residual = encoder(image, message)
        """

        encoded_image, effective_residual, _ = (
            self.generate_trigger_with_mask(image, message)
        )

        return encoded_image, effective_residual

    def forward_with_mask(self, image, message):
        """
        Diagnostic forward method.

        Use this later for visualizing where the hider places residual.
        """

        return self.generate_trigger_with_mask(image, message)


# ============================================================
# Spatial transformer for decoder
# ============================================================

class SpatialTransformer(nn.Module):
    """
    Lightweight spatial transformer.

    Original StegaStamp uses a spatial transformer in the decoder to improve
    robustness against alignment changes.

    This CIFAR-10 version predicts a small affine transformation and is
    initialized as identity.
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

    Receives encoded or corrupted image and predicts hidden message bits.

    Output:
        raw logits of shape (B, message_length)

    During training:
        use BCEWithLogitsLoss.
    """

    def __init__(
        self,
        message_length=16,
        base_channels=32,
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

    Included for future perceptual-quality extension.
    Not needed for the first stable CA3 implementation.
    """

    def __init__(self, base_channels=32):
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
        -> advanced encoder / hider
        -> encoded image
        -> differentiable corruption layer
        -> decoder
        -> decoded message logits

    Inference / trigger generation flow:
        clean image + fixed target message
        -> encoder
        -> encoded image
    """

    def __init__(
        self,
        message_length=16,
        image_size=32,
        message_map_size=8,
        message_channels=8,
        base_channels=32,
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

        encoded_image, residual, residual_mask = (
            self.encoder.forward_with_mask(image, message)
        )

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
            "residual_mask": residual_mask,
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
    def encode_with_mask(self, image, message):
        """
        Encode a message and also return the adaptive residual mask.
        """

        self.eval()

        check_image_tensor(image)
        check_message_tensor(message, self.message_length)

        encoded_image, residual, residual_mask = (
            self.encoder.forward_with_mask(image, message)
        )

        return encoded_image, residual, residual_mask

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
        message_length=16,
        image_size=32,
        message_map_size=8,
        message_channels=8,
        base_channels=32,
        max_residual=8 / 255,
        use_spatial_transformer=True,
        use_corruption=True
    ).to(device)

    images = torch.rand(4, 3, 32, 32).to(device)

    messages = create_random_message(
        batch_size=4,
        message_length=16,
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
    residual_masks = outputs["residual_mask"]
    corrupted_images = outputs["corrupted_image"]
    decoded_logits = outputs["decoded_logits"]

    acc = bit_accuracy(decoded_logits, messages)
    psnr = compute_psnr(encoded_images, images)
    residual_stats = compute_residual_stats(residuals)

    print("Input images:", images.shape)
    print("Encoded images:", encoded_images.shape)
    print("Residuals:", residuals.shape)
    print("Residual masks:", residual_masks.shape)
    print("Corrupted images:", corrupted_images.shape)
    print("Decoded logits:", decoded_logits.shape)
    print("Initial bit accuracy:", acc)
    print("Initial PSNR:", psnr)
    print("Residual stats:", residual_stats)

    print("Encoded image min:", encoded_images.min().item())
    print("Encoded image max:", encoded_images.max().item())
    print("Residual max abs:", residuals.abs().max().item())
    print("Residual bound:", model.max_residual)
    print("Mask min:", residual_masks.min().item())
    print("Mask max:", residual_masks.max().item())

    assert encoded_images.shape == images.shape
    assert residuals.shape == images.shape
    assert residual_masks.shape == (4, 1, 32, 32)
    assert corrupted_images.shape == images.shape
    assert decoded_logits.shape == (4, 16)
    assert encoded_images.min().item() >= 0.0
    assert encoded_images.max().item() <= 1.0
    assert residuals.abs().max().item() <= model.max_residual + 1e-6

    print("Advanced stegastamp_like_model.py sanity check passed.")