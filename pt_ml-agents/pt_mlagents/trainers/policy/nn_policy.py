from typing import Any, Dict, Optional, List
from pt_mlagents.pt_utils import torch
from pt_mlagents_envs.timers import timed
from pt_mlagents_envs.base_env import DecisionSteps, BehaviorSpec
from pt_mlagents.trainers.models import EncoderType
from pt_mlagents.trainers.models import ModelUtils
from pt_mlagents.trainers.policy.pt_policy import PTPolicy
from pt_mlagents.trainers.settings import TrainerSettings
from pt_mlagents.trainers.distributions import (
    GaussianDistribution,
    MultiCategoricalDistribution,
)

EPSILON = 1e-6  # Small value to avoid divide by zero


class NNPolicy(PTPolicy):
    def __init__(
            self,
            seed: int,
            behavior_spec: BehaviorSpec,
            trainer_params: TrainerSettings,
            is_training: bool,
            model_path: str,
            load: bool,
            tanh_squash: bool = False,
            reparameterize: bool = False,
            condition_sigma_on_obs: bool = True,
            create_pt_graph: bool = True,
    ):
        """
        Policy that uses a multilayer perceptron to map the observations to actions. Could
        also use a CNN to encode visual input prior to the MLP. Supports discrete and
        continuous action spaces, as well as recurrent networks.
        :param seed: Random seed.
        :param brain: Assigned BrainParameters object.
        :param trainer_params: Defined training parameters.
        :param is_training: Whether the model should be trained.
        :param load: Whether a pre-trained model will be loaded or a new one created.
        :param model_path: Path where the model should be saved and loaded.
        :param tanh_squash: Whether to use a tanh function on the continuous output, or a clipped output.
        :param reparameterize: Whether we are using the resampling trick to update the policy in continuous output.
        """
        super().__init__(seed, behavior_spec, trainer_params, model_path, load)
        self.grads = None
        self.update_batch: Optional[torch.Operation] = None
        num_layers = self.network_settings.num_layers
        self.h_size = self.network_settings.hidden_units
        if num_layers < 1:
            num_layers = 1
        self.num_layers = num_layers
        self.vis_encode_type = self.network_settings.vis_encode_type
        self.tanh_squash = tanh_squash
        self.reparameterize = reparameterize
        self.condition_sigma_on_obs = condition_sigma_on_obs
        self.trainable_variables: List[torch.Variable] = []

        # Non-exposed parameters; these aren't exposed because they don't have a
        # good explanation and usually shouldn't be touched.
        self.log_std_min = -20
        self.log_std_max = 2
        if create_pt_graph:
            self.create_pt_graph()

    def get_trainable_variables(self) -> List[torch.Tensor]:
        """
        Returns a List of the trainable variables in this policy. if create_pt_graph hasn't been called,
        returns empty list.
        """
        return self.trainable_variables

    def create_pt_graph(self) -> None:
        """
        Builds the pytorch graph needed for this policy.
        """
        # with self.graph.as_default():
        torch.random.manual_seed(self.seed)

        # FIXME : Get all global parameters in pytorch.
        # _vars = torch.get_collection(torch.GraphKeys.GLOBAL_VARIABLES)
        _vars = []
        if len(_vars) > 0:
            # We assume the first thing created in the graph is the Policy. If
            # already populated, don't create more tensors.
            return

        # FIXME : I think no input placeholder is needed for pytorch version.
        # self.create_input_placeholders()
        encoded = self._create_encoder(
            [],  # self.visual_in,
            torch.FloatTensor(size=(1, 8)),  # self.processed_vector_in,
            self.h_size,
            self.num_layers,
            self.vis_encode_type,
        )
        if self.use_continuous_act:
            self._create_cc_actor(
                encoded,
                self.tanh_squash,
                self.reparameterize,
                self.condition_sigma_on_obs,
            )
        else:
            self._create_dc_actor(encoded)
        self.trainable_variables = torch.get_collection(
            torch.GraphKeys.TRAINABLE_VARIABLES, scope="policy"
        )
        self.trainable_variables += torch.get_collection(
            torch.GraphKeys.TRAINABLE_VARIABLES, scope="lstm"
        )  # LSTMs need to be root scope for Barracuda export

        self.inference_dict: Dict[str, torch.Tensor] = {
            "action": self.output,
            "log_probs": self.all_log_probs,
            "entropy": self.entropy,
        }
        if self.use_continuous_act:
            self.inference_dict["pre_action"] = self.output_pre
        if self.use_recurrent:
            self.inference_dict["memory_out"] = self.memory_out

        # We do an initialize to make the Policy usable out of the box. If an optimizer is needed,
        # it will re-load the full graph
        self._initialize_graph()

    @timed
    def evaluate(
            self, decision_requests: DecisionSteps, global_agent_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluates policy for the agent experiences provided.
        :param decision_requests: DecisionSteps object containing inputs.
        :param global_agent_ids: The global (with worker ID) agent ids of the data in the batched_step_result.
        :return: Outputs from network as defined by self.inference_dict.
        """
        feed_dict = {
            self.batch_size_ph: len(decision_requests),
            self.sequence_length_ph: 1,
        }
        if self.use_recurrent:
            if not self.use_continuous_act:
                feed_dict[self.prev_action] = self.retrieve_previous_action(
                    global_agent_ids
                )
            feed_dict[self.memory_in] = self.retrieve_memories(global_agent_ids)
        feed_dict = self.fill_eval_dict(feed_dict, decision_requests)
        run_out = self._execute_model(feed_dict, self.inference_dict)
        return run_out

    def _create_encoder(
            self,
            visual_in: List[torch.Tensor],
            vector_in: torch.Tensor,
            h_size: int,
            num_layers: int,
            vis_encode_type: EncoderType,
    ) -> torch.nn.Module:
        """
        Creates an encoder for visual and vector observations.
        :param h_size: Size of hidden linear layers.
        :param num_layers: Number of hidden linear layers.
        :param vis_encode_type: Type of visual encoder to use if visual input.
        :return: The hidden layer (torch.Tensor) after the encoder.
        """
        for shape in self.behavior_spec.observation_shapes:
            print(f"shape : {shape}")

        return ModelUtils.create_observation_streams(
            visual_in,
            shape,
            1,
            h_size,
            num_layers,
            vis_encode_type,
        )[0]

    def _create_cc_actor(
            self,
            encoded: torch.nn.ModuleList,
            tanh_squash: bool = False,
            reparameterize: bool = False,
            condition_sigma_on_obs: bool = True,
    ) -> torch.nn.Sequential:
        """
        Creates Continuous control actor-critic model.
        :param h_size: Size of hidden linear layers.
        :param num_layers: Number of hidden linear layers.
        :param vis_encode_type: Type of visual encoder to use if visual input.
        :param tanh_squash: Whether to use a tanh function, or a clipped output.
        :param reparameterize: Whether we are using the resampling trick to update the policy.
        """

        modules = torch.nn.ModuleList()

        if self.use_recurrent:
            self.memory_in = torch.placeholder(
                shape=[None, self.m_size], dtype=torch.float32, name="recurrent_in"
            )
            hidden_policy, memory_policy_out = ModelUtils.create_recurrent_encoder(
                encoded, self.memory_in, self.sequence_length_ph, name="lstm_policy"
            )

            self.memory_out = torch.identity(memory_policy_out, name="recurrent_out")
        else:
            hidden_policy = encoded

        # with torch.variable_scope("policy"):
        distribution = GaussianDistribution(
            hidden_policy,
            self.act_size,
            reparameterize=reparameterize,
            tanh_squash=tanh_squash,
            condition_sigma=condition_sigma_on_obs,
        )
        modules.add_module("gaussian_distribution", distribution)

        if not tanh_squash:
            self.output_pre = distribution.sample
            # Clip and scale output to ensure actions are always within [-1, 1] range.
            output_post = torch.clamp(self.output_pre, -3, 3) / 3
            self.output = torch.Tensor(output_post, name="action")
            modules.add_module("clamp", torch.nn.Module())

        self.selected_actions = self.output.detach()

        self.all_log_probs = torch.Tensor(distribution.log_probs, name="action_probs")
        self.entropy = distribution.entropy

        # We keep these tensors the same name, but use new nodes to keep code parallelism with discrete control.
        self.total_log_probs = distribution.total_log_probs

    def _create_dc_actor(self, encoded: torch.Tensor) -> None:
        """
        Creates Discrete control actor-critic model.
        :param h_size: Size of hidden linear layers.
        :param num_layers: Number of hidden linear layers.
        :param vis_encode_type: Type of visual encoder to use if visual input.
        """
        if self.use_recurrent:
            self.prev_action = torch.placeholder(
                shape=[None, len(self.act_size)], dtype=torch.int32, name="prev_action"
            )
            prev_action_oh = torch.concat(
                [
                    torch.one_hot(self.prev_action[:, i], self.act_size[i])
                    for i in range(len(self.act_size))
                ],
                axis=1,
            )
            hidden_policy = torch.concat([encoded, prev_action_oh], axis=1)

            self.memory_in = torch.placeholder(
                shape=[None, self.m_size], dtype=torch.float32, name="recurrent_in"
            )
            hidden_policy, memory_policy_out = ModelUtils.create_recurrent_encoder(
                hidden_policy,
                self.memory_in,
                self.sequence_length_ph,
                name="lstm_policy",
            )

            self.memory_out = torch.identity(memory_policy_out, "recurrent_out")
        else:
            hidden_policy = encoded

        self.action_masks = torch.placeholder(
            shape=[None, sum(self.act_size)], dtype=torch.float32, name="action_masks"
        )

        with torch.variable_scope("policy"):
            distribution = MultiCategoricalDistribution(
                hidden_policy, self.act_size, self.action_masks
            )
        # It's important that we are able to feed_dict a value into this tensor to get the
        # right one-hot encoding, so we can't do identity on it.
        self.output = distribution.sample
        self.all_log_probs = torch.identity(distribution.log_probs, name="action")
        self.selected_actions = torch.stop_gradient(
            distribution.sample_onehot
        )  # In discrete, these are onehot
        self.entropy = distribution.entropy
        self.total_log_probs = distribution.total_log_probs
