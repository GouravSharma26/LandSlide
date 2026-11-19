"""Convolutional LSTM (ConvLSTM) implementation for spatio-temporal modeling.

Provides ConvLSTMCell and multi-layer ConvLSTM with sequence masking
for variable-length observation sequences and missing observations.
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Single-layer 2D Convolutional LSTM cell.

    Args:
        in_channels: Number of input channels.
        hidden_channels: Number of hidden state channels.
        kernel_size: Size of convolutional kernel (default 3).
        bias: Whether to include bias parameters (default True).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # 4 gates: input (i), forget (f), cell candidate (g), output (o)
        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Step the ConvLSTM cell for a single time step.

        Args:
            x: Input tensor of shape [B, C, H, W]
            state: Tuple (h_prev, c_prev) of shape [B, H_c, H, W].
                   If None, initialized to zeros.
            valid_mask: Optional boolean or binary mask of shape [B] or [B, 1, 1, 1].
                        If 0/False, state is preserved unchanged (missing observation).

        Returns:
            Tuple of (h_next, c_next) of shape [B, H_c, H, W]
        """
        batch_size, _, height, width = x.shape

        if state is None:
            h_prev = torch.zeros(
                batch_size, self.hidden_channels, height, width,
                device=x.device, dtype=x.dtype
            )
            c_prev = torch.zeros(
                batch_size, self.hidden_channels, height, width,
                device=x.device, dtype=x.dtype
            )
        else:
            h_prev, c_prev = state

        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)

        # Split into 4 gates
        i_gate, f_gate, g_gate, o_gate = torch.chunk(gates, chunks=4, dim=1)

        i_t = torch.sigmoid(i_gate)
        f_t = torch.sigmoid(f_gate)
        g_t = torch.tanh(g_gate)
        o_t = torch.sigmoid(o_gate)

        c_next = f_t * c_prev + i_t * g_t
        h_next = o_t * torch.tanh(c_next)

        # Apply validity mask if provided (missing observation handling)
        if valid_mask is not None:
            # Reshape or interpolate valid_mask to match [B, 1, height, width]
            if valid_mask.ndim == 1:
                mask_4d = valid_mask.view(batch_size, 1, 1, 1).float()
            elif valid_mask.ndim == 2:
                mask_4d = valid_mask.view(batch_size, 1, 1, 1).float()
            else:
                mask_4d = valid_mask.float()
                if mask_4d.shape[-2:] != (height, width):
                    mask_4d = torch.nn.functional.interpolate(
                        mask_4d, size=(height, width), mode="nearest"
                    )

            # Where mask == 0, keep previous states (no update)
            c_next = mask_4d * c_next + (1.0 - mask_4d) * c_prev
            h_next = mask_4d * h_next + (1.0 - mask_4d) * h_prev

        return h_next, c_next


class ConvLSTM(nn.Module):
    """Multi-layer Convolutional LSTM accepting 5D spatio-temporal tensors [B, T, C, H, W].

    Args:
        in_channels: Number of input channels.
        hidden_channels: Hidden dimension per layer (int or list of ints).
        kernel_size: Convolution kernel size (default 3).
        num_layers: Number of stacked ConvLSTM layers (default 1).
        batch_first: If True, input format is [B, T, C, H, W] (default True).
        bias: Whether to use bias in convolutions.
        return_all_layers: If True, returns hidden states for all stacked layers.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Union[int, List[int]],
        kernel_size: int = 3,
        num_layers: int = 1,
        batch_first: bool = True,
        bias: bool = True,
        return_all_layers: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        if isinstance(hidden_channels, int):
            hidden_list = [hidden_channels] * num_layers
        else:
            if len(hidden_channels) != num_layers:
                raise ValueError(
                    f"hidden_channels list length ({len(hidden_channels)}) must match num_layers ({num_layers})"
                )
            hidden_list = hidden_channels
        self.hidden_channels_list = hidden_list

        cell_list = []
        for i in range(num_layers):
            cur_in = in_channels if i == 0 else hidden_list[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    in_channels=cur_in,
                    hidden_channels=hidden_list[i],
                    kernel_size=kernel_size,
                    bias=bias,
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        valid_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass across temporal sequence.

        Args:
            x: 5D tensor [B, T, C, H, W] (if batch_first=True)
            hidden_state: Optional list of (h, c) tuples for each layer.
            valid_masks: Optional sequence validity mask:
                         - [B, T]: boolean / binary per time step
                         - [B, T, 1, H, W] / [B, T, 1, 1, 1]: spatial mask

        Returns:
            layer_output: [B, T, hidden_channels[-1], H, W]
            last_states: List of (h, c) states for each layer at final time step
        """
        if not self.batch_first:
            # (T, B, C, H, W) -> (B, T, C, H, W)
            x = x.permute(1, 0, 2, 3, 4)

        batch_size, time_steps, _, height, width = x.shape

        if hidden_state is None:
            hidden_state = [None] * self.num_layers

        cur_layer_input = x
        all_layer_outputs = []
        last_states = []

        for layer_idx, cell in enumerate(self.cell_list):
            h, c = hidden_state[layer_idx] if hidden_state[layer_idx] is not None else (None, None)
            output_inner = []

            for t in range(time_steps):
                x_t = cur_layer_input[:, t, :, :, :]
                mask_t = None
                if valid_masks is not None:
                    if valid_masks.ndim == 2:  # [B, T]
                        mask_t = valid_masks[:, t]
                    elif valid_masks.ndim == 5:  # [B, T, 1, H, W]
                        mask_t = valid_masks[:, t, :, :, :]

                h, c = cell(x=x_t, state=(h, c) if h is not None else None, valid_mask=mask_t)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)  # [B, T, H_c, H, W]
            cur_layer_input = layer_output
            all_layer_outputs.append(layer_output)
            last_states.append((h, c))

        if self.return_all_layers:
            return all_layer_outputs, last_states
        return cur_layer_input, last_states
