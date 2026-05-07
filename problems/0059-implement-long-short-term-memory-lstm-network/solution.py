import torch
from torch import mm
from torch import sigmoid
from torch import mm
from torch import cat
from torch import tanh

class LSTM:
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases as float64 tensors
        self.Wf = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)
        self.Wi = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)
        self.Wc = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)
        self.Wo = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)

        self.bf = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bi = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bc = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bo = torch.zeros(hidden_size, 1, dtype=torch.float64)

    def forward(self, x: torch.Tensor, initial_hidden_state: torch.Tensor, initial_cell_state: torch.Tensor):
        """
        Processes a sequence of inputs and returns the hidden states,
        final hidden state, and final cell state.

        Args:
            x: Input tensor of shape (seq_len, input_size)
            initial_hidden_state: Initial hidden state of shape (hidden_size, 1)
            initial_cell_state: Initial cell state of shape (hidden_size, 1)

        Returns:
            outputs: Tensor of hidden states at each time step
            h: Final hidden state tensor
            c: Final cell state tensor
        """
        seq_len,input_size = x.shape
        outputs = []
        h_t,c_t = initial_hidden_state,initial_cell_state
        for t in range(seq_len):
            xt = x[t].reshape((input_size,1))
            # print(initial_hidden_state,xt)
            f_t = sigmoid(mm(self.Wf,cat((h_t,xt),dim=0))+self.bf)
            i_t = sigmoid(mm(self.Wi,cat((h_t,xt),dim=0))+self.bi)
            c_ht = tanh(mm(self.Wc,cat((h_t,xt),dim=0))+self.bc)
            c_t = f_t * c_t + i_t * c_ht
            o_t = sigmoid(mm(self.Wo,cat((h_t,xt),dim=0))+self.bo)
            h_t = o_t * tanh(c_t)
            outputs.append(h_t)

        return torch.stack(outputs, dim=0),h_t,c_t