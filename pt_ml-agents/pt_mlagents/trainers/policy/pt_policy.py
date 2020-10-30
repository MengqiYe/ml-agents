from typing import Any, Dict, List, Optional, Tuple
import abc
import os
import numpy as np
from distutils.version import LooseVersion

from pt_mlagents.pt_utils import torch
from pt_mlagents import pt_utils
from pt_mlagents_envs.exception import UnityException
from pt_mlagents_envs.base_env import BehaviorSpec
from pt_mlagents_envs.logging_util import get_logger
from pt_mlagents.trainers.policy import Policy
from pt_mlagents.trainers.action_info import ActionInfo
from pt_mlagents.trainers.trajectory import SplitObservations
from pt_mlagents.trainers.behavior_id_utils import get_global_agent_id
from pt_mlagents_envs.base_env import DecisionSteps
from pt_mlagents.trainers.models import ModelUtils
from pt_mlagents.trainers.settings import TrainerSettings, NetworkSettings
from pt_mlagents.trainers import __version__


logger = get_logger(__name__)


# This is the version number of the inputs and outputs of the model, and
# determines compatibility with inference in Barracuda.
MODEL_FORMAT_VERSION = 2


class UnityPolicyException(UnityException):
    """
    Related to errors with the Trainer.
    """

    pass


class PTPolicy(Policy):
    """
    Contains a learning model, and the necessary
    functions to save/load models and create the input placeholders.
    """

    def __init__(
        self,
        seed: int,
        behavior_spec: BehaviorSpec,
        trainer_settings: TrainerSettings,
        model_path: str,
        load: bool = False,
    ):
        """
        Initialized the policy.
        :param seed: Random seed to use for TensorFlow.
        :param brain: The corresponding Brain for this policy.
        :param trainer_settings: The trainer parameters.
        :param model_path: Where to load/save the model.
        :param load: If True, load model from model_path. Otherwise, create new model.
        """

        self.m_size = 0
        self.trainer_settings = trainer_settings
        self.network_settings: NetworkSettings = trainer_settings.network_settings
        # for ghost trainer save/load snapshots
        self.assign_phs: List[torch.Tensor] = []
        self.assign_ops: List[torch.Operation] = []

        self.inference_dict: Dict[str, torch.Tensor] = {}
        self.update_dict: Dict[str, torch.Tensor] = {}
        self.sequence_length = 1
        self.seed = seed
        self.behavior_spec = behavior_spec

        self.act_size = (
            list(behavior_spec.discrete_action_branches)
            if behavior_spec.is_action_discrete()
            else [behavior_spec.action_size]
        )
        self.vec_obs_size = sum(
            shape[0] for shape in behavior_spec.observation_shapes if len(shape) == 1
        )
        self.vis_obs_size = sum(
            1 for shape in behavior_spec.observation_shapes if len(shape) == 3
        )

        self.use_recurrent = self.network_settings.memory is not None
        self.memory_dict: Dict[str, np.ndarray] = {}
        self.num_branches = self.behavior_spec.action_size
        self.previous_action_dict: Dict[str, np.array] = {}
        self.normalize = self.network_settings.normalize
        self.use_continuous_act = behavior_spec.is_action_continuous()
        self.model_path = model_path
        self.initialize_path = self.trainer_settings.init_path
        self.keep_checkpoints = self.trainer_settings.keep_checkpoints
        self.graph = torch.Graph()
        # self.sess = torch.Session(
        #     config=pt_utils.generate_session_config(), graph=self.graph
        # )
        self.saver: Optional[torch.Operation] = None
        self.seed = seed
        if self.network_settings.memory is not None:
            self.m_size = self.network_settings.memory.memory_size
            self.sequence_length = self.network_settings.memory.sequence_length
        self._initialize_pytorch_references()
        self.load = load

    @abc.abstractmethod
    def get_trainable_variables(self) -> List[torch.Tensor]:
        """
        Returns a List of the trainable variables in this policy. if create_pt_graph hasn't been called,
        returns empty list.
        """
        pass

    @abc.abstractmethod
    def create_pt_graph(self):
        """
        Builds the pytorch graph needed for this policy.
        """
        pass

    @staticmethod
    def _convert_version_string(version_string: str) -> Tuple[int, ...]:
        """
        Converts the version string into a Tuple of ints (major_ver, minor_ver, patch_ver).
        :param version_string: The semantic-versioned version string (X.Y.Z).
        :return: A Tuple containing (major_ver, minor_ver, patch_ver).
        """
        ver = LooseVersion(version_string)
        return tuple(map(int, ver.version[0:3]))

    def _check_model_version(self, version: str) -> None:
        """
        Checks whether the model being loaded was created with the same version of
        ML-Agents, and throw a warning if not so.
        """
        if self.version_tensors is not None:
            loaded_ver = tuple(
                num.eval(session=self.sess) for num in self.version_tensors
            )
            if loaded_ver != PTPolicy._convert_version_string(version):
                logger.warning(
                    f"The model checkpoint you are loading from was saved with ML-Agents version "
                    f"{loaded_ver[0]}.{loaded_ver[1]}.{loaded_ver[2]} but your current ML-Agents"
                    f"version is {version}. Model may not behave properly."
                )

    def _initialize_graph(self):
        with self.graph.as_default():
            self.saver = torch.train.Saver(max_to_keep=self.keep_checkpoints)
            init = torch.global_variables_initializer()
            self.sess.run(init)

    def _load_graph(self, model_path: str, reset_global_steps: bool = False) -> None:
        with self.graph.as_default():
            self.saver = torch.train.Saver(max_to_keep=self.keep_checkpoints)
            logger.info(f"Loading model from {model_path}.")
            ckpt = torch.train.get_checkpoint_state(model_path)
            if ckpt is None:
                raise UnityPolicyException(
                    "The model {0} could not be loaded. Make "
                    "sure you specified the right "
                    "--run-id and that the previous run you are loading from had the same "
                    "behavior names.".format(model_path)
                )
            try:
                self.saver.restore(self.sess, cktorch.model_checkpoint_path)
            except torch.errors.NotFoundError:
                raise UnityPolicyException(
                    "The model {0} was found but could not be loaded. Make "
                    "sure the model is from the same version of ML-Agents, has the same behavior parameters, "
                    "and is using the same trainer configuration as the current run.".format(
                        model_path
                    )
                )
            self._check_model_version(__version__)
            if reset_global_steps:
                self._set_step(0)
                logger.info(
                    "Starting training from step 0 and saving to {}.".format(
                        self.model_path
                    )
                )
            else:
                logger.info(
                    "Resuming training from step {}.".format(self.get_current_step())
                )

    def initialize_or_load(self):
        # If there is an initialize path, load from that. Else, load from the set model path.
        # If load is set to True, don't reset steps to 0. Else, do. This allows a user to,
        # e.g., resume from an initialize path.
        reset_steps = not self.load
        if self.initialize_path is not None:
            self._load_graph(self.initialize_path, reset_global_steps=reset_steps)
        elif self.load:
            self._load_graph(self.model_path, reset_global_steps=reset_steps)
        else:
            self._initialize_graph()

    def get_weights(self):
        with self.graph.as_default():
            _vars = torch.get_collection(torch.GraphKeys.GLOBAL_VARIABLES)
            values = [v.eval(session=self.sess) for v in _vars]
            return values

    def init_load_weights(self):
        with self.graph.as_default():
            _vars = torch.get_collection(torch.GraphKeys.GLOBAL_VARIABLES)
            values = [v.eval(session=self.sess) for v in _vars]
            for var, value in zip(_vars, values):
                assign_ph = torch.placeholder(var.dtype, shape=value.shape)
                self.assign_phs.append(assign_ph)
                self.assign_ops.append(torch.assign(var, assign_ph))

    def load_weights(self, values):
        if len(self.assign_ops) == 0:
            logger.warning(
                "Calling load_weights in pt_policy but assign_ops is empty. Did you forget to call init_load_weights?"
            )
        with self.graph.as_default():
            feed_dict = {}
            for assign_ph, value in zip(self.assign_phs, values):
                feed_dict[assign_ph] = value
            self.sess.run(self.assign_ops, feed_dict=feed_dict)

    def evaluate(
        self, decision_requests: DecisionSteps, global_agent_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluates policy for the agent experiences provided.
        :param decision_requests: DecisionSteps input to network.
        :return: Output from policy based on self.inference_dict.
        """
        raise UnityPolicyException("The evaluate function was not implemented.")

    def get_action(
        self, decision_requests: DecisionSteps, worker_id: int = 0
    ) -> ActionInfo:
        """
        Decides actions given observations information, and takes them in environment.
        :param decision_requests: A dictionary of brain names and DecisionSteps from environment.
        :param worker_id: In parallel environment training, the unique id of the environment worker that
            the DecisionSteps came from. Used to construct a globally unique id for each agent.
        :return: an ActionInfo containing action, memories, values and an object
        to be passed to add experiences
        """
        if len(decision_requests) == 0:
            return ActionInfo.empty()

        global_agent_ids = [
            get_global_agent_id(worker_id, int(agent_id))
            for agent_id in decision_requests.agent_id
        ]  # For 1-D array, the iterator order is correct.

        run_out = self.evaluate(  # pylint: disable=assignment-from-no-return
            decision_requests, global_agent_ids
        )

        self.save_memories(global_agent_ids, run_out.get("memory_out"))
        return ActionInfo(
            action=run_out.get("action"),
            value=run_out.get("value"),
            outputs=run_out,
            agent_ids=decision_requests.agent_id,
        )

    def update(self, mini_batch, num_sequences):
        """
        Performs update of the policy.
        :param num_sequences: Number of experience trajectories in batch.
        :param mini_batch: Batch of experiences.
        :return: Results of update.
        """
        raise UnityPolicyException("The update function was not implemented.")

    def _execute_model(self, feed_dict, out_dict):
        """
        Executes model.
        :param feed_dict: Input dictionary mapping nodes to input data.
        :param out_dict: Output dictionary mapping names to nodes.
        :return: Dictionary mapping names to input data.
        """
        network_out = self.sess.run(list(out_dict.values()), feed_dict=feed_dict)
        run_out = dict(zip(list(out_dict.keys()), network_out))
        return run_out

    def fill_eval_dict(self, feed_dict, batched_step_result):
        vec_vis_obs = SplitObservations.from_observations(batched_step_result.obs)
        for i, _ in enumerate(vec_vis_obs.visual_observations):
            feed_dict[self.visual_in[i]] = vec_vis_obs.visual_observations[i]
        if self.use_vec_obs:
            feed_dict[self.vector_in] = vec_vis_obs.vector_observations
        if not self.use_continuous_act:
            mask = np.ones(
                (
                    len(batched_step_result),
                    sum(self.behavior_spec.discrete_action_branches),
                ),
                dtype=np.float32,
            )
            if batched_step_result.action_mask is not None:
                mask = 1 - np.concatenate(batched_step_result.action_mask, axis=1)
            feed_dict[self.action_masks] = mask
        return feed_dict

    def make_empty_memory(self, num_agents):
        """
        Creates empty memory for use with RNNs
        :param num_agents: Number of agents.
        :return: Numpy array of zeros.
        """
        return np.zeros((num_agents, self.m_size), dtype=np.float32)

    def save_memories(
        self, agent_ids: List[str], memory_matrix: Optional[np.ndarray]
    ) -> None:
        if memory_matrix is None:
            return
        for index, agent_id in enumerate(agent_ids):
            self.memory_dict[agent_id] = memory_matrix[index, :]

    def retrieve_memories(self, agent_ids: List[str]) -> np.ndarray:
        memory_matrix = np.zeros((len(agent_ids), self.m_size), dtype=np.float32)
        for index, agent_id in enumerate(agent_ids):
            if agent_id in self.memory_dict:
                memory_matrix[index, :] = self.memory_dict[agent_id]
        return memory_matrix

    def remove_memories(self, agent_ids):
        for agent_id in agent_ids:
            if agent_id in self.memory_dict:
                self.memory_dict.pop(agent_id)

    def make_empty_previous_action(self, num_agents):
        """
        Creates empty previous action for use with RNNs and discrete control
        :param num_agents: Number of agents.
        :return: Numpy array of zeros.
        """
        return np.zeros((num_agents, self.num_branches), dtype=np.int)

    def save_previous_action(
        self, agent_ids: List[str], action_matrix: Optional[np.ndarray]
    ) -> None:
        if action_matrix is None:
            return
        for index, agent_id in enumerate(agent_ids):
            self.previous_action_dict[agent_id] = action_matrix[index, :]

    def retrieve_previous_action(self, agent_ids: List[str]) -> np.ndarray:
        action_matrix = np.zeros((len(agent_ids), self.num_branches), dtype=np.int)
        for index, agent_id in enumerate(agent_ids):
            if agent_id in self.previous_action_dict:
                action_matrix[index, :] = self.previous_action_dict[agent_id]
        return action_matrix

    def remove_previous_action(self, agent_ids):
        for agent_id in agent_ids:
            if agent_id in self.previous_action_dict:
                self.previous_action_dict.pop(agent_id)

    def get_current_step(self):
        """
        Gets current model step.
        :return: current model step.
        """
        step = self.sess.run(self.global_step)
        return step

    def _set_step(self, step: int) -> int:
        """
        Sets current model step to step without creating additional ops.
        :param step: Step to set the current model step to.
        :return: The step the model was set to.
        """
        current_step = self.get_current_step()
        # Increment a positive or negative number of steps.
        return self.increment_step(step - current_step)

    def increment_step(self, n_steps):
        """
        Increments model step.
        """
        out_dict = {
            "global_step": self.global_step,
            "increment_step": self.increment_step_op,
        }
        feed_dict = {self.steps_to_increment: n_steps}
        return self.sess.run(out_dict, feed_dict=feed_dict)["global_step"]

    def get_inference_vars(self):
        """
        :return:list of inference var names
        """
        return list(self.inference_dict.keys())

    def get_update_vars(self):
        """
        :return:list of update var names
        """
        return list(self.update_dict.keys())

    def save_model(self, steps):
        """
        Saves the model
        :param steps: The number of steps the model was trained for
        :return:
        """
        with self.graph.as_default():
            last_checkpoint = os.path.join(self.model_path, f"model-{steps}.ckpt")
            self.saver.save(self.sess, last_checkpoint)
            torch.train.write_graph(
                self.graph, self.model_path, "raw_graph_def.pb", as_text=False
            )

    def update_normalization(self, vector_obs: np.ndarray) -> None:
        """
        If this policy normalizes vector observations, this will update the norm values in the graph.
        :param vector_obs: The vector observations to add to the running estimate of the distribution.
        """
        if self.use_vec_obs and self.normalize:
            self.sess.run(
                self.update_normalization_op, feed_dict={self.vector_in: vector_obs}
            )

    @property
    def use_vis_obs(self):
        return self.vis_obs_size > 0

    @property
    def use_vec_obs(self):
        return self.vec_obs_size > 0

    def _initialize_pytorch_references(self):
        self.value_heads: Dict[str, torch.Tensor] = {}
        self.normalization_steps: Optional[torch.Variable] = None
        self.running_mean: Optional[torch.Variable] = None
        self.running_variance: Optional[torch.Variable] = None
        self.update_normalization_op: Optional[torch.Operation] = None
        self.value: Optional[torch.Tensor] = None
        self.all_log_probs: torch.Tensor = None
        self.total_log_probs: Optional[torch.Tensor] = None
        self.entropy: Optional[torch.Tensor] = None
        self.output_pre: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.selected_actions: torch.Tensor = None
        self.action_masks: Optional[torch.Tensor] = None
        self.prev_action: Optional[torch.Tensor] = None
        self.memory_in: Optional[torch.Tensor] = None
        self.memory_out: Optional[torch.Tensor] = None
        self.version_tensors: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

    def create_input_placeholders(self):
        # with self.graph.as_default():
        (
            self.global_step,
            self.increment_step_op,
            self.steps_to_increment,
        ) = ModelUtils.create_global_steps()
        self.vector_in, self.visual_in = ModelUtils.create_input_placeholders(
            self.behavior_spec.observation_shapes
        )
        if self.normalize:
            normalization_tensors = ModelUtils.create_normalizer(self.vector_in)
            self.update_normalization_op = normalization_tensors.update_op
            self.normalization_steps = normalization_tensors.steps
            self.running_mean = normalization_tensors.running_mean
            self.running_variance = normalization_tensors.running_variance
            self.processed_vector_in = ModelUtils.normalize_vector_obs(
                self.vector_in,
                self.running_mean,
                self.running_variance,
                self.normalization_steps,
            )
        else:
            self.processed_vector_in = self.vector_in
            self.update_normalization_op = None

        self.batch_size_ph = torch.placeholder(
            shape=None, dtype=torch.int32, name="batch_size"
        )
        self.sequence_length_ph = torch.placeholder(
            shape=None, dtype=torch.int32, name="sequence_length"
        )
        self.mask_input = torch.placeholder(
            shape=[None], dtype=torch.float32, name="masks"
        )
        # Only needed for PPO, but needed for BC module
        self.epsilon = torch.placeholder(
            shape=[None, self.act_size[0]], dtype=torch.float32, name="epsilon"
        )
        self.mask = torch.cast(self.mask_input, torch.int32)

        torch.Variable(
            int(self.behavior_spec.is_action_continuous()),
            name="is_continuous_control",
            trainable=False,
            dtype=torch.int32,
        )
        int_version = PTPolicy._convert_version_string(__version__)
        major_ver_t = torch.Variable(
            int_version[0],
            name="trainer_major_version",
            trainable=False,
            dtype=torch.int32,
        )
        minor_ver_t = torch.Variable(
            int_version[1],
            name="trainer_minor_version",
            trainable=False,
            dtype=torch.int32,
        )
        patch_ver_t = torch.Variable(
            int_version[2],
            name="trainer_patch_version",
            trainable=False,
            dtype=torch.int32,
        )
        self.version_tensors = (major_ver_t, minor_ver_t, patch_ver_t)
        torch.Variable(
            MODEL_FORMAT_VERSION,
            name="version_number",
            trainable=False,
            dtype=torch.int32,
        )
        torch.Variable(
            self.m_size, name="memory_size", trainable=False, dtype=torch.int32
        )
        if self.behavior_spec.is_action_continuous():
            torch.Variable(
                self.act_size[0],
                name="action_output_shape",
                trainable=False,
                dtype=torch.int32,
            )
        else:
            torch.Variable(
                sum(self.act_size),
                name="action_output_shape",
                trainable=False,
                dtype=torch.int32,
            )
