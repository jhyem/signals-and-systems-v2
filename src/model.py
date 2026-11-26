import torch
import torch.nn as nn


class MaskCNN(nn.Module):
    """
    Baseline MaskCNN for speech denoising.
    Input: waveform [B, T]
    Output: enhanced waveform [B, T]
    """

    def __init__(
        self,
        n_fft=512,
        hop_length=128,
        hidden_channels=(64, 64),
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.register_buffer(
            "window",
            torch.hann_window(n_fft)
        )

        self.network = nn.Sequential(
            nn.Conv1d(
                n_fft // 2 + 1,
                hidden_channels[0],
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(),

            nn.Conv1d(
                hidden_channels[0],
                hidden_channels[1],
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv1d(
                hidden_channels[1],
                n_fft // 2 + 1,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, noisy_waveform):
        """
        noisy_waveform: [B, T]
        """

        window = self.window.to(noisy_waveform.device)

        noisy_stft = torch.stft(
            noisy_waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )

        magnitude = noisy_stft.abs()

        features = torch.log1p(magnitude)

        mask = self.network(features)

        enhanced_stft = noisy_stft * mask

        enhanced_waveform = torch.istft(
            enhanced_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            length=noisy_waveform.shape[-1],
        )

        return enhanced_waveform
