from keras import saving, ops, layers
import keras
from tensorflow.keras import ops


@saving.register_keras_serializable()
class TimeDistributedDenseBlock(layers.Layer):
    """Time-Distributed Fully Connected layer block."""

    def __init__(self, bottleneck_factor, fft_dim, **kwargs):
        super().__init__(**kwargs)
        self.fft_dim = fft_dim
        self.hidden_dim = fft_dim // bottleneck_factor
        self.bottleneck_factor = bottleneck_factor

    def build(self, *_):
        self.group_norm_1 = layers.GroupNormalization(groups=-1)
        self.group_norm_2 = layers.GroupNormalization(groups=-1)
        self.dense_1 = layers.Dense(self.hidden_dim, use_bias=False)
        self.dense_2 = layers.Dense(self.fft_dim, use_bias=False)

    def call(self, x):
        x = ops.gelu(self.group_norm_1(x))
        x = ops.swapaxes(x, -1, -2)
        x = self.dense_1(x)

        x = ops.gelu(self.group_norm_2(ops.swapaxes(x, -1, -2)))
        x = ops.swapaxes(x, -1, -2)
        x = self.dense_2(x)
        return ops.swapaxes(x, -1, -2)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "bottleneck_factor": self.bottleneck_factor,
                "fft_dim": self.fft_dim,
            }
        )
        return config


@saving.register_keras_serializable()
class TimeFrequencyConvolution(layers.Layer):
    """Time-Frequency Convolutional layer."""

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels

    def build(self, *_):
        self.group_norm = layers.GroupNormalization(groups=-1)
        self.conv = layers.Conv2D(self.channels, 3, padding="same", use_bias=False)

    def call(self, x):
        return self.conv(ops.gelu(self.group_norm(x)))

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels})
        return config


@saving.register_keras_serializable()
class TimeFrequencyTransformBlock(layers.Layer):
    """Implements TFC_TDF block for encoder-decoder architecture."""

    def __init__(
        self, channels, length, fft_dim, bottleneck_factor, in_channels=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.length = length
        self.fft_dim = fft_dim
        self.bottleneck_factor = bottleneck_factor
        self.in_channels = in_channels or channels
        self.blocks = []

    def build(self, *_):
        for i in range(self.length):
            in_channels = self.channels if i > 0 else self.in_channels
            self.blocks.append(TimeFrequencyConvolution(in_channels))
            self.blocks.append(
                TimeDistributedDenseBlock(self.bottleneck_factor, self.fft_dim)
            )
            self.blocks.append(TimeFrequencyConvolution(self.channels))
            self.blocks.append(layers.Conv2D(self.channels, 1, 1, use_bias=False))

    def call(self, inputs):
        x = inputs
        for i in range(0, len(self.blocks), 4):
            tfc_1 = self.blocks[i](x)
            tdf = self.blocks[i + 1](x)
            tfc_2 = self.blocks[i + 2](tfc_1 + tdf)
            x = tfc_2 + self.blocks[i + 3](x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "length": self.length,
                "fft_dim": self.fft_dim,
                "bottleneck_factor": self.bottleneck_factor,
                "in_channels": self.in_channels,
            }
        )
        return config


@saving.register_keras_serializable()
class Downscale(layers.Layer):
    """Downscale time-frequency dimensions using a convolution."""

    def __init__(self, channels, scale, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.scale = scale

    def build(self, *_):
        self.conv = layers.Conv2D(self.channels, self.scale, self.scale, use_bias=False)
        self.norm = layers.GroupNormalization(groups=-1)

    def call(self, inputs):
        return self.norm(ops.gelu(self.conv(inputs)))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "scale": self.scale,
            }
        )
        return config


@saving.register_keras_serializable()
class Upscale(layers.Layer):
    """Upscale time-frequency dimensions using a transposed convolution."""

    def __init__(self, channels, scale, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.scale = scale

    def build(self, *_):
        self.conv = layers.Conv2DTranspose(
            self.channels, self.scale, self.scale, use_bias=False
        )
        self.norm = layers.GroupNormalization(groups=-1)

    def call(self, inputs):
        return self.norm(ops.gelu(self.conv(inputs)))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "scale": self.scale,
            }
        )
        return config
