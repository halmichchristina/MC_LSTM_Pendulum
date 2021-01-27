import torch
import torch.nn as nn
import numpy as np

from models.normalisers import NormalisedSigmoid
from models.redistributions import Gate, get_redistribution
from experiment.pendulum.utils import choose_nonlinearity

class NoInputMassConserving(nn.Module):
    """ Mass Conserving LSTMs with no input """
    def __init__(self,
                 hidden_size: int,
                 hidden_layer_size: int = 100, 
                 redistribution_type: str = "pendulum",
                 normaliser: str = "softmax",
                 batch_first: bool = True, 
                 initial_output_bias: float = None,
                 scale_c: bool = True,
                 friction: bool = False):
        """
                    
        Parameters
        ----------
        mass_input_size : int
            Number of mass input features at each time step.
        aux_input_size : int
            Number of auxiliary input features at each time step.
        hidden_size : int
            Number of output features at each time step.
        redistribution_type : str, optional
            Specifies how the redistribution matrix should be computed.
        batch_first : bool, optional
            Whether or not the first dimension is the batch dimension.
        """
        super(NoInputMassConserving, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layer_size = hidden_layer_size
        self.redistribution_type = redistribution_type
        self.initial_output_bias = initial_output_bias
        self.scale_c = scale_c
        self.batch_first = batch_first

        self.friction = friction

        if normaliser == 'sigmoid':
            self.normaliser = NormalisedSigmoid(dim=-1)
        else:
            self.normaliser = nn.Softmax(dim=-1)
        self.out_gate = Gate(self.hidden_size, self.hidden_size)
        # NOTE: without normalised sigmoid here, there seem to be troubles!
        self.redistribution = get_redistribution(self.redistribution_type,
                                                 num_states=self.hidden_size,
                                                 num_features=100,
                                                 hidden_layer_size = self.hidden_layer_size,
                                                 normaliser=self.normaliser)
        self.out_gate.reset_parameters()
        self.redistribution.reset_parameters()
        self.reset_parameters()

        self.embedder = nn.Sequential(
            nn.Linear(self.hidden_size + 9, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU()
        )


        self.fc_state  = nn.Linear(64, hidden_size)



    def reset_parameters(self):
        if(self.initial_output_bias is not None):
            nn.init.constant_(self.out_gate.fc.bias, val= self.initial_output_bias)

    def forward(self, init_state: torch.Tensor,
                n_time_steps: int,
                ct_v: torch.Tensor = None,
                ct_a: torch.Tensor = None,
                xa: torch.Tensor = None,
                ):# -> Tuple[torch.Tensor, torch.Tensor]:
        ct = init_state

        m_out, c, r = [], [],[]
        c.append(ct)

        for t in range(n_time_steps):
            hz = torch.cat([ct, xa[:, t]], dim=1)
            conc = self.embedder(hz)
            mt_out, ct, rt = self._step(ct, conc) 
            m_out.append(mt_out)
            c.append(1.0002*ct - 0.0001)   #D: important so that softmax does not go to extremes
            r.append(torch.diagonal(torch.fliplr(rt)[0])) #antidiagonal of r for plot
        m_out, c, r = torch.stack(m_out), torch.stack(c), torch.stack(r)
        r = r.unsqueeze(1)
        if self.batch_first:
            m_out = m_out.transpose(0, 1)
            c = c.transpose(0, 1)
            r = r.transpose(0, 1)
        return m_out, c,r

    def _step(self, ct: torch.Tensor, conc: torch.Tensor):# -> Tuple[torch.Tensor, torch.Tensor]:
        """ Make a single time step in the MCLSTM. """
        #if self.scale_c:

        r = self.redistribution(conc)
        c_out = torch.matmul(ct.unsqueeze(-2), r).squeeze(-2)
        mt_out = c_out  #D: just a placeholder for debugging
        if self.friction:
            o = self.out_gate(ct)
            c_out = (1 - o) * c_out
            mt_out = o * c_out
        return mt_out,  c_out, r
class JustAnARLSTM(nn.Module):
    """ Mass Conserving LSTMs with no input """

    def __init__(self,
                     lstm_hidden: int = 256,
                 batch_first: bool = True):

        super(JustAnARLSTM, self).__init__()
        self.batch_first = batch_first
        self.lstm_hidden = lstm_hidden
        self.c_lstm_init = torch.zeros(1,self.lstm_hidden)
        self.h_lstm_init = torch.zeros(1,self.lstm_hidden)

        self.lstm_cell = nn.LSTMCell(11, self.lstm_hidden)
        self.fc = nn.Linear(self.lstm_hidden, 2)

    def reset_parameters(self):
        if (self.initial_output_bias is not None):
            nn.init.constant_(self.out_gate.fc.bias, val=self.initial_output_bias)

    def forward(self, init_state: torch.Tensor,
                n_time_steps: int,
                xa: torch.Tensor = None,
                ):  # -> Tuple[torch.Tensor, torch.Tensor]:
        ct = init_state
        c_lstm = self.c_lstm_init
        h_lstm = self.h_lstm_init

        m_out, c = [], []
        c.append(ct)

        for t in range(n_time_steps):
            lstm_in = torch.cat([ct, xa[:, t]], dim=1)
            h_lstm, c_lstm = self.lstm_cell(lstm_in, (h_lstm, c_lstm))
            ct = self.fc(c_lstm)
            m_out.append(h_lstm)
            c.append(ct)

        m_out, c = torch.stack(m_out), torch.stack(c)
        if self.batch_first:
            m_out = m_out.transpose(0, 1)
            c = c.transpose(0, 1)
        return m_out, c
class HNN(torch.nn.Module):
    '''Learn arbitrary vector fields that are sums of conservative and solenoidal fields'''
    def __init__(self, input_dim, differentiable_model, field_type='solenoidal',
                    baseline=False, assume_canonical_coords=True):
        super(HNN, self).__init__()
        self.baseline = baseline
        self.differentiable_model = differentiable_model
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(input_dim) # Levi-Civita permutation tensor
        self.field_type = field_type

    def forward(self, x):
        # traditional forward pass
        if self.baseline:
            return self.differentiable_model(x)

        y = self.differentiable_model(x)
        assert y.dim() == 2 and y.shape[1] == 2, "Output tensor should have shape [batch_size, 2]"
        return y.split(1,1)

    def rk4_time_derivative(self, x, dt):
        return rk4(fun=self.time_derivative, y0=x, t=0, dt=dt)

    def time_derivative(self, x, t=None, separate_fields=False):
        '''NEURAL ODE-STLE VECTOR FIELD'''
        if self.baseline:
            return self.differentiable_model(x)

        '''NEURAL HAMILTONIAN-STLE VECTOR FIELD'''
        F1, F2 = self.forward(x) # traditional forward pass

        conservative_field = torch.zeros_like(x) # start out with both components set to 0
        solenoidal_field = torch.zeros_like(x)

        if self.field_type != 'solenoidal':
            dF1 = torch.autograd.grad(F1.sum(), x, create_graph=True)[0] # gradients for conservative field
            conservative_field = dF1 @ torch.eye(*self.M.shape)

        if self.field_type != 'conservative':
            dF2 = torch.autograd.grad(F2.sum(), x, create_graph=True)[0] # gradients for solenoidal field
            solenoidal_field = dF2 @ self.M.t()

        if separate_fields:
            return [conservative_field, solenoidal_field]

        return conservative_field + solenoidal_field

    def permutation_tensor(self,n):
        M = None
        if self.assume_canonical_coords:
            M = torch.eye(n)
            M = torch.cat([M[n//2:], -M[:n//2]])
        else:
            '''Constructs the Levi-Civita permutation tensor'''
            M = torch.ones(n,n) # matrix of ones
            M *= 1 - torch.eye(n) # clear diagonals
            M[::2] *= -1 # pattern of signs
            M[:,::2] *= -1
    
            for i in range(n): # make asymmetric
                for j in range(i+1, n):
                    M[i,j] *= -1
        return M
class MLP(torch.nn.Module):
  '''Just a salt-of-the-earth MLP'''
  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh'):
    super(MLP, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=None)

    for l in [self.linear1, self.linear2, self.linear3]:
      torch.nn.init.orthogonal_(l.weight) # use a principled initialization

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def forward(self, x, separate_fields=False):
    h = self.nonlinearity( self.linear1(x) )
    h = self.nonlinearity( self.linear2(h) )
    return self.linear3(h)
