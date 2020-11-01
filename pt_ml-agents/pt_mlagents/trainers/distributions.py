import abc
from typing import NamedTuple, List, Tuple
import numpy as np

from pt_mlagents.pt_utils import torch
from pt_mlagents.trainers.models import ModelUtils, Route, ClampExp

EPSILON = 1e-6  # Small value to avoid divide by zero


class OutputDistribution(abc.ABC):
    @abc.abstractproperty
    def log_probs(self) -> torch.Tensor:
        """
        Returns a Tensor that when evaluated, produces the per-action log probabilities of this distribution.
        The shape of this Tensor should be equivalent to (batch_size x the number of actions) produced in sample.
        """
        pass

    @abc.abstractproperty
    def total_log_probs(self) -> torch.Tensor:
        """
        Returns a Tensor that when evaluated, produces the total log probability for a single sample.
        The shape of this Tensor should be equivalent to (batch_size x 1) produced in sample.
        """
        pass

    @abc.abstractproperty
    def sample(self) -> torch.Tensor:
        """
        Returns a Tensor that when evaluated, produces a sample of this OutputDistribution.
        """
        pass

    @abc.abstractproperty
    def entropy(self) -> torch.Tensor:
        """
        Returns a Tensor that when evaluated, produces the entropy of this distribution.
        """
        pass


class DiscreteOutputDistribution(OutputDistribution):
    @abc.abstractproperty
    def sample_onehot(self) -> torch.Tensor:
        """
        Returns a one-hot version of the output.
        """


class GaussianDistribution(OutputDistribution, torch.nn.ModuleList):
    """
    A Gaussian output distribution for continuous actions.
    """

    class MuSigmaTensors(NamedTuple):
        mu: torch.Tensor
        log_sigma: torch.Tensor
        sigma: torch.Tensor

    class MuLogSigmaModule(torch.nn.Module):
        def __init__(
                self,
                logits: torch.Tensor,
                act_size: List[int],
                log_sigma_min: float,
                log_sigma_max: float,
                condition_sigma: bool,
        ):
            super(GaussianDistribution.MuLogSigmaModule, self).__init__()
            self.mu_linear = torch.nn.Linear(in_features=1, out_features=act_size[0])
            self.log_sigma_min = log_sigma_min
            self.log_sigma_max = log_sigma_max

        def forward(self, x):
            mu = self.mu_linear(x)
            log_sigma = x.clamp(self.log_sigma_min, self.log_sigma_max)
            sigma = log_sigma.exp()
            return self.MuSigmaTensors(mu, log_sigma, sigma)

    class SamplePolicyModule(torch.nn.Module):
        def __init__(self, parent):
            super(GaussianDistribution.SamplePolicyModule, self).__init__()
            self.parent: GaussianDistribution = parent
            self.normal = torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))

        def forward(self, args, **kwargs):
            epsilon = self.normal
            self.parent._sampled_policy = self.parent.encoded.mu + self.parent.encoded.sigma * epsilon
            return self.parent._sampled_policy

    class LogProbModule(torch.nn.Module):
        def __init__(self, parent, sampled_policy_probs):
            super(GaussianDistribution.LogProbModule, self).__init__()
            self.parent: GaussianDistribution = parent
            self.sample = sampled_policy_probs

        def forward(self):
            _gauss_pre = -0.5 * (
                    ((self.sample - self.parent.encoded.mu) / (self.parent.encoded.sigma + EPSILON)) ** 2
                    + 2 * self.parent.encoded.log_sigma
                    + np.log(2 * np.pi)
            )
            return _gauss_pre

    class TanhSquashModule(torch.nn.Module):
        def __init__(self, parent):
            super(GaussianDistribution.TanhSquashModule, self).__init__()
            self.parent: GaussianDistribution = parent

        def forward(self, args, **kwargs):
            self.parent.sample = torch.tanh(self.parent.sample)
            self.parent.total_log_probs = self._do_squash_correction_for_tanh(
                self._all_probs, self._sampled_policy
            )

        def _do_squash_correction_for_tanh(self, probs, squashed_policy) -> torch.Tensor:
            adjusted_probs = probs - torch.log(1 - squashed_policy ** 2 + EPSILON)
            return adjusted_probs

    class UpdateTotalLogProbs(torch.nn.Module):
        def __init__(self, parent):
            super(GaussianDistribution.UpdateTotalLogProbs, self).__init__()
            self.parent = parent

        def forward(self, x):
            self.parent._total_prob = torch.sum(self.parent._all_probs, dim=1, keepdim=True)
            return self.parent.total_log_probs

    class EntropyModule(torch.nn.Module):
        def __init__(self, parent):
            super(GaussianDistribution.EntropyModule, self).__init__()
            self.parent: GaussianDistribution = parent

        def forward(self, x):
            single_dim_entropy = 0.5 * torch.reduce_mean(
                torch.log(2 * np.pi * np.e) + 2 * self.parent.encoded.log_sigma
            )
            ones = torch.ones_like(torch.reshape(self.parent.encoded.mu[:, 0], [-1]))
            self.parent._entropy = ones * single_dim_entropy

    def __init__(
            self,
            # logits: torch.Tensor,
            act_size: List[int],
            reparameterize: bool = False,
            tanh_squash: bool = False,
            condition_sigma: bool = True,
            log_sigma_min: float = -20,
            log_sigma_max: float = 2,
    ) -> torch.nn.Module:
        """
        A Gaussian output distribution for continuous actions.
        :param logits: Hidden layer to use as the input to the Gaussian distribution.
        :param act_size: List containing the number of continuous actions.
        :param reparameterize: Whether or not to use the reparameterization trick (block gradients through
            log probability calculation.)
        :param tanh_squash: Squash the output using tanh, constraining it between -1 and 1.
            From: Haarnoja et. al, https://arxiv.org/abs/1801.01290
        :param log_sigma_min: Minimum log standard deviation to clip by.
        :param log_sigma_max: Maximum log standard deviation to clip by.
        """
        super(GaussianDistribution, self).__init__()
        self._encoded = None
        self._sampled_policy = None
        self._sampled_policy_probs = None

        self.add_module("mu_log_sigma", GaussianDistribution.MuLogSigmaModule(
            logits,
            act_size,
            log_sigma_min,
            log_sigma_max,
            condition_sigma=condition_sigma,
        ))
        self.add_module("sample_policy", self.SamplePolicyModule(self))

        if not reparameterize:
            _sampled_policy_probs = self._sampled_policy # .detach()
        else:
            _sampled_policy_probs = self._sampled_policy

        # self._all_probs = self._create_log_probs(_sampled_policy_probs, encoded)
        self.add_module("log_prob", self.LogProbModule(self, _sampled_policy_probs))
        if tanh_squash:
            self.add_module("tanh_squash", tanh_squash())
        self.add_module("total_log_probs", GaussianDistribution.UpdateTotalLogProbs(self))
        self.add_module("entropy", GaussianDistribution.EntropyModule(self))


    def _create_log_probs(
            self, sampled_policy: torch.Tensor, encoded: "GaussianDistribution.MuSigmaTensors"
    ) -> torch.Tensor:
        _gauss_pre = -0.5 * (
                ((sampled_policy - encoded.mu) / (encoded.sigma + EPSILON)) ** 2
                + 2 * encoded.log_sigma
                + np.log(2 * np.pi)
        )
        return _gauss_pre

    def _create_entropy(
            self, encoded: "GaussianDistribution.MuSigmaTensors"
    ) -> torch.Tensor:
        single_dim_entropy = 0.5 * torch.reduce_mean(
            torch.log(2 * np.pi * np.e) + 2 * encoded.log_sigma
        )
        # Make entropy the right shape
        return torch.ones_like(torch.reshape(encoded.mu[:, 0], [-1])) * single_dim_entropy

    def _do_squash_correction_for_tanh(self, probs, squashed_policy):
        """
        Adjust probabilities for squashed sample before output
        """
        adjusted_probs = probs - torch.log(1 - squashed_policy ** 2 + EPSILON)
        return adjusted_probs

    @property
    def encoded(self) -> None:
        return self._encoded

    @property
    def total_log_probs(self) -> torch.Tensor:
        return self._total_prob

    @property
    def log_probs(self) -> torch.Tensor:
        return self._all_probs

    @property
    def sample(self) -> torch.Tensor:
        return self._sampled_policy

    @property
    def entropy(self) -> torch.Tensor:
        return self._entropy


class MultiCategoricalDistribution(DiscreteOutputDistribution):
    """
    A categorical distribution for multi-branched discrete actions. Also supports action masking.
    """

    def __init__(self, logits: torch.Tensor, act_size: List[int], action_masks: torch.Tensor):
        """
        A categorical distribution for multi-branched discrete actions.
        :param logits: Hidden layer to use as the input to the Gaussian distribution.
        :param act_size: List containing the number of discrete actions per branch.
        :param action_masks: Tensor representing action masks. Should be of length sum(act_size), and 0 for masked
            and 1 for unmasked.
        """
        unmasked_log_probs = self._create_policy_branches(logits, act_size)
        (
            self._sampled_policy,
            self._all_probs,
            action_index,
        ) = self._get_masked_actions_probs(unmasked_log_probs, act_size, action_masks)
        self._sampled_onehot = self._action_onehot(self._sampled_policy, act_size)
        self._entropy = self._create_entropy(self._all_probs, action_index, act_size)
        self._total_prob = self._get_log_probs(
            self._sampled_onehot, self._all_probs, action_index, act_size
        )

    def _create_policy_branches(
            self, logits: torch.Tensor, act_size: List[int]
    ) -> List[torch.Tensor]:
        policy_branches = []
        for size in act_size:
            policy_branches.append(
                torch.layers.dense(
                    logits,
                    size,
                    activation=None,
                    use_bias=False,
                    kernel_initializer=ModelUtils.scaled_init(0.01),
                )
            )
        return policy_branches

    def _get_masked_actions_probs(
            self,
            unmasked_log_probs: List[torch.Tensor],
            act_size: List[int],
            action_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        output, _, all_log_probs = ModelUtils.create_discrete_action_masking_layer(
            unmasked_log_probs, action_masks, act_size
        )

        action_idx = [0] + list(np.cumsum(act_size))
        return output, all_log_probs, action_idx

    def _action_onehot(self, sample: torch.Tensor, act_size: List[int]) -> torch.Tensor:
        action_oh = torch.concat(
            [torch.one_hot(sample[:, i], act_size[i]) for i in range(len(act_size))],
            axis=1,
        )
        return action_oh

    def _get_log_probs(
            self,
            sample_onehot: torch.Tensor,
            all_log_probs: torch.Tensor,
            action_idx: List[int],
            act_size: List[int],
    ) -> torch.Tensor:
        log_probs = torch.reduce_sum(
            (
                torch.stack(
                    [
                        -torch.nn.softmax_cross_entropy_with_logits_v2(
                            labels=sample_onehot[:, action_idx[i]: action_idx[i + 1]],
                            logits=all_log_probs[:, action_idx[i]: action_idx[i + 1]],
                        )
                        for i in range(len(act_size))
                    ],
                    axis=1,
                )
            ),
            axis=1,
            keepdims=True,
        )
        return log_probs

    def _create_entropy(
            self, all_log_probs: torch.Tensor, action_idx: List[int], act_size: List[int]
    ) -> torch.Tensor:
        entropy = torch.reduce_sum(
            (
                torch.stack(
                    [
                        torch.nn.softmax_cross_entropy_with_logits_v2(
                            labels=torch.nn.softmax(
                                all_log_probs[:, action_idx[i]: action_idx[i + 1]]
                            ),
                            logits=all_log_probs[:, action_idx[i]: action_idx[i + 1]],
                        )
                        for i in range(len(act_size))
                    ],
                    axis=1,
                )
            ),
            axis=1,
        )

        return entropy

    @property
    def log_probs(self) -> torch.Tensor:
        return self._all_probs

    @property
    def total_log_probs(self) -> torch.Tensor:
        return self._total_prob

    @property
    def sample(self) -> torch.Tensor:
        return self._sampled_policy

    @property
    def sample_onehot(self) -> torch.Tensor:
        return self._sampled_onehot

    @property
    def entropy(self) -> torch.Tensor:
        return self._entropy
