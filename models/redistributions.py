import torch
from torch import nn

from models.normalisers import NormalisedSigmoid


def get_redistribution(kind: str,
                       num_states: int,
                       num_features: int = None,
                       num_out: int = None,
                       normaliser: nn.Module = None,
                       **kwargs):
    if kind == "pendulum":
        return PendulumRedistribution(num_states, num_features, kwargs['hidden_layer_size'], num_out, normaliser)
    else:
        raise ValueError("unknown kind of redistribution: {}".format(kind))


class Redistribution(nn.Module):
    """ Base class for modules that generate redistribution vectors/matrices. """

    def __init__(self, num_states: int, num_features: int = None, num_out: int = None, normaliser: nn.Module = None):
        """
        Parameters
        ----------
        num_states : int
            The number of states this redistribution is to be applied on.
        num_features : int, optional
            The number of features to use for configuring the redistribution.
            If the redistribution is not input-dependent, this argument will be ignored.
        num_out : int, optional
            The number of outputs to redistribute the states to.
            If nothing is specified, the redistribution matrix is assumed to be square.
        normaliser : Module, optional
            Function to use for normalising the redistribution matrix.
        """
        super().__init__()
        self.num_features = num_features
        self.num_states = num_states
        self.num_out = num_out or num_states
        self.normaliser = normaliser or NormalisedSigmoid(dim=-1)

    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("subclass must implement this method")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self._compute(x)
        return self.normaliser(r)


class Gate(Redistribution):
    """
    Classic gate as used in e.g. LSTMs.
    
    Notes
    -----
    The vector that is computed by this module gives rise to a diagonal redistribution matrix,
    i.e. a redistribution matrix that does not really redistribute (not normalised).
    """

    def __init__(self, num_states, num_features, num_out=None, sigmoid=None):
        super().__init__(num_states, num_features, 1, sigmoid or nn.Sigmoid())
        self.fc = nn.Linear(num_features, num_states)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class PendulumRedistribution(Redistribution):
    """
    Gate-like redistribution that only depends on input.
    
    This module directly computes all entries for the redistribution matrix
    from a linear combination of the input values and is normalised by the activation function.
    """

    """
    Parameters
    ----------
    num_states : int
        The number of states this redistribution is to be applied on.
    num_features : int, optional
        The number of features to use for configuring the redistribution.
        If the redistribution is not input-dependent, this argument will be ignored.
    num_out : int, optional
        The number of outputs to redistribute the states to.
        If nothing is specified, the redistribution matrix is assumed to be square.
    normaliser : Module, optional
        Function to use for normalising the redistribution matrix.
    """

    def __init__(self, num_states, num_features, hidden_layer_size, num_out=None, normaliser=None):
        super().__init__(num_states, num_features, num_out, normaliser)
        #self.fc_1 = nn.Linear(num_features, hidden_layer_size)
        #self.fc_2 = nn.Linear(hidden_layer_size, self.num_states * self.num_out)
        self.fc = nn.Linear(num_features, self.num_states * self.num_out )
        self.reset_parameters()

    def reset_parameters(self):
        # TODO: account for effect normaliser
        #nn.init.orthogonal_(self.fc_1.weight)
        #nn.init.zeros_(self.fc_1.bias)
        #nn.init.orthogonal_(self.fc_2.weight)
        #nn.init.zeros_(self.fc_2.bias)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.uniform_(self.fc.bias)

    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        #logits = self.fc_2(torch.relu(self.fc_1(x)))
        logits = self.fc(x)
        return logits.view(-1, self.num_states, self.num_out)
