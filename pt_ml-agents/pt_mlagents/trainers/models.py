from enum import Enum
from typing import Callable, Dict, List, Tuple, NamedTuple

import numpy as np
from pt_mlagents.pt_utils import pt

from pt_mlagents.trainers.exception import UnityTrainerException

ActivationFunction = Callable[[pt.Tensor], pt.Tensor]
EncoderFunction = Callable[
    [pt.Tensor, int, ActivationFunction, int, str, bool], pt.Tensor
]

EPSILON = 1e-7


class Tensor3DShape(NamedTuple):
    height: int
    width: int
    num_channels: int


class EncoderType(Enum):
    SIMPLE = "simple"
    NATURE_CNN = "nature_cnn"
    RESNET = "resnet"


class ScheduleType(Enum):
    CONSTANT = "constant"
    LINEAR = "linear"


class NormalizerTensors(NamedTuple):
    # FIXME: Find what is Operation in pytorch.
    update_op: None # pt.Operation
    steps: pt.Tensor
    running_mean: pt.Tensor
    running_variance: pt.Tensor


class ModelUtils:
    # Minimum supported side for each encoder type. If refactoring an encoder, please
    # adjust these also.
    MIN_RESOLUTION_FOR_ENCODER = {
        EncoderType.SIMPLE: 20,
        EncoderType.NATURE_CNN: 36,
        EncoderType.RESNET: 15,
    }

    @staticmethod
    def create_global_steps():
        """Creates PT ops to track and increment global training step."""
        global_step = pt.Variable(
            0, name="global_step", trainable=False, dtype=pt.int32
        )
        steps_to_increment = pt.placeholder(
            shape=[], dtype=pt.int32, name="steps_to_increment"
        )
        increment_step = pt.assign(global_step, pt.add(global_step, steps_to_increment))
        return global_step, increment_step, steps_to_increment

    @staticmethod
    def create_schedule(
        schedule: ScheduleType,
        parameter: float,
        global_step: pt.Tensor,
        max_step: int,
        min_value: float,
    ) -> pt.Tensor:
        """
        Create a learning rate tensor.
        :param lr_schedule: Type of learning rate schedule.
        :param lr: Base learning rate.
        :param global_step: A PT Tensor representing the total global step.
        :param max_step: The maximum number of steps in the training run.
        :return: A Tensor containing the learning rate.
        """
        if schedule == ScheduleType.CONSTANT:
            parameter_rate = pt.Variable(parameter, trainable=False)
        elif schedule == ScheduleType.LINEAR:
            parameter_rate = pt.train.polynomial_decay(
                parameter, global_step, max_step, min_value, power=1.0
            )
        else:
            raise UnityTrainerException("The schedule {} is invalid.".format(schedule))
        return parameter_rate

    @staticmethod
    def scaled_init(scale):
        return pt.initializers.variance_scaling(scale)

    @staticmethod
    def swish(input_activation: pt.Tensor) -> pt.Tensor:
        """Swish activation function. For more info: https://arxiv.org/abs/1710.05941"""
        return pt.multiply(input_activation, pt.nn.sigmoid(input_activation))

    @staticmethod
    def create_visual_input(camera_parameters: Tensor3DShape, name: str) -> pt.Tensor:
        """
        Creates image input op.
        :param camera_parameters: Parameters for visual observation.
        :param name: Desired name of input op.
        :return: input op.
        """
        o_size_h = camera_parameters.height
        o_size_w = camera_parameters.width
        c_channels = camera_parameters.num_channels

        visual_in = pt.placeholder(
            shape=[None, o_size_h, o_size_w, c_channels], dtype=pt.float32, name=name
        )
        return visual_in

    @staticmethod
    def create_input_placeholders(
        observation_shapes: List[Tuple], name_prefix: str = ""
    ) -> Tuple[pt.Tensor, List[pt.Tensor]]:
        """
        Creates input placeholders for visual inputs.
        :param observation_shapes: A List of tuples that specify the resolutions
            of the input observations. Tuples for now are restricted to 1D (vector) or 3D (Tensor)
        :param name_prefix: A name prefix to add to the placeholder names. This is used so that there
            is no conflict when creating multiple placeholder sets.
        :returns: A List of Tensorflow placeholders where the input iamges should be fed.
        """
        visual_in: List[pt.Tensor] = []
        vector_in_size = 0
        for i, dimension in enumerate(observation_shapes):
            if len(dimension) == 3:
                _res = Tensor3DShape(
                    height=dimension[0], width=dimension[1], num_channels=dimension[2]
                )
                visual_input = ModelUtils.create_visual_input(
                    _res, name=name_prefix + "visual_observation_" + str(i)
                )
                visual_in.append(visual_input)
            elif len(dimension) == 1:
                vector_in_size += dimension[0]
            else:
                raise UnityTrainerException(
                    f"Unsupported shape of {dimension} for observation {i}"
                )
        vector_in = pt.placeholder(
            shape=[None, vector_in_size],
            dtype=pt.float32,
            name=name_prefix + "vector_observation",
        )
        return vector_in, visual_in

    @staticmethod
    def create_vector_input(
        vec_obs_size: int, name: str = "vector_observation"
    ) -> pt.Tensor:
        """
        Creates ops for vector observation input.
        :param vec_obs_size: Size of stacked vector observation.
        :param name: Name of the placeholder op.
        :return: Placeholder for vector observations.
        """
        vector_in = pt.placeholder(
            shape=[None, vec_obs_size], dtype=pt.float32, name=name
        )
        return vector_in

    @staticmethod
    def normalize_vector_obs(
        vector_obs: pt.Tensor,
        running_mean: pt.Tensor,
        running_variance: pt.Tensor,
        normalization_steps: pt.Tensor,
    ) -> pt.Tensor:
        """
        Create a normalized version of an input tensor.
        :param vector_obs: Input vector observation tensor.
        :param running_mean: Tensorflow tensor representing the current running mean.
        :param running_variance: Tensorflow tensor representing the current running variance.
        :param normalization_steps: Tensorflow tensor representing the current number of normalization_steps.
        :return: A normalized version of vector_obs.
        """
        normalized_state = pt.clip_by_value(
            (vector_obs - running_mean)
            / pt.sqrt(
                running_variance / (pt.cast(normalization_steps, pt.float32) + 1)
            ),
            -5,
            5,
            name="normalized_state",
        )
        return normalized_state

    @staticmethod
    def create_normalizer(vector_obs: pt.Tensor) -> NormalizerTensors:
        """
        Creates the normalizer and the variables required to store its state.
        :param vector_obs: A Tensor representing the next value to normalize. When the
            update operation is called, it will use vector_obs to update the running mean
            and variance.
        :return: A NormalizerTensors tuple that holds running mean, running variance, number of steps,
            and the update operation.
        """

        vec_obs_size = vector_obs.shape[1]
        steps = pt.get_variable(
            "normalization_steps",
            [],
            trainable=False,
            dtype=pt.int32,
            initializer=pt.zeros_initializer(),
        )
        running_mean = pt.get_variable(
            "running_mean",
            [vec_obs_size],
            trainable=False,
            dtype=pt.float32,
            initializer=pt.zeros_initializer(),
        )
        running_variance = pt.get_variable(
            "running_variance",
            [vec_obs_size],
            trainable=False,
            dtype=pt.float32,
            initializer=pt.ones_initializer(),
        )
        update_normalization = ModelUtils.create_normalizer_update(
            vector_obs, steps, running_mean, running_variance
        )
        return NormalizerTensors(
            update_normalization, steps, running_mean, running_variance
        )

    @staticmethod
    def create_normalizer_update(
        vector_input: pt.Tensor,
        steps: pt.Tensor,
        running_mean: pt.Tensor,
        running_variance: pt.Tensor,
    ) -> None: # pt.Operation:
        """
        Creates the update operation for the normalizer.
        :param vector_input: Vector observation to use for updating the running mean and variance.
        :param running_mean: Tensorflow tensor representing the current running mean.
        :param running_variance: Tensorflow tensor representing the current running variance.
        :param steps: Tensorflow tensor representing the current number of steps that have been normalized.
        :return: A PT operation that updates the normalization based on vector_input.
        """
        # Based on Welford's algorithm for running mean and standard deviation, for batch updates. Discussion here:
        # https://stackoverflow.com/questions/56402955/whats-the-formula-for-welfords-algorithm-for-variance-std-with-batch-updates
        steps_increment = pt.shape(vector_input)[0]
        total_new_steps = pt.add(steps, steps_increment)

        # Compute the incremental update and divide by the number of new steps.
        input_to_old_mean = pt.subtract(vector_input, running_mean)
        new_mean = running_mean + pt.reduce_sum(
            input_to_old_mean / pt.cast(total_new_steps, dtype=pt.float32), axis=0
        )
        # Compute difference of input to the new mean for Welford update
        input_to_new_mean = pt.subtract(vector_input, new_mean)
        new_variance = running_variance + pt.reduce_sum(
            input_to_new_mean * input_to_old_mean, axis=0
        )
        update_mean = pt.assign(running_mean, new_mean)
        update_variance = pt.assign(running_variance, new_variance)
        update_norm_step = pt.assign(steps, total_new_steps)
        return pt.group([update_mean, update_variance, update_norm_step])

    @staticmethod
    def create_vector_observation_encoder(
        observation_input: pt.Tensor,
        h_size: int,
        activation: ActivationFunction,
        num_layers: int,
        scope: str,
        reuse: bool,
    ) -> pt.Tensor:
        """
        Builds a set of hidden state encoders.
        :param reuse: Whether to re-use the weights within the same scope.
        :param scope: Graph scope for the encoder ops.
        :param observation_input: Input vector.
        :param h_size: Hidden layer size.
        :param activation: What type of activation function to use for layers.
        :param num_layers: number of hidden layers to create.
        :return: List of hidden layer tensors.
        """
        with pt.variable_scope(scope):
            hidden = observation_input
            for i in range(num_layers):
                hidden = pt.layers.dense(
                    hidden,
                    h_size,
                    activation=activation,
                    reuse=reuse,
                    name="hidden_{}".format(i),
                    kernel_initializer=pt.initializers.variance_scaling(1.0),
                )
        return hidden

    @staticmethod
    def create_visual_observation_encoder(
        image_input: pt.Tensor,
        h_size: int,
        activation: ActivationFunction,
        num_layers: int,
        scope: str,
        reuse: bool,
    ) -> pt.Tensor:
        """
        Builds a set of resnet visual encoders.
        :param image_input: The placeholder for the image input to use.
        :param h_size: Hidden layer size.
        :param activation: What type of activation function to use for layers.
        :param num_layers: number of hidden layers to create.
        :param scope: The scope of the graph within which to create the ops.
        :param reuse: Whether to re-use the weights within the same scope.
        :return: List of hidden layer tensors.
        """
        with pt.variable_scope(scope):
            conv1 = pt.layers.conv2d(
                image_input,
                16,
                kernel_size=[8, 8],
                strides=[4, 4],
                activation=pt.nn.elu,
                reuse=reuse,
                name="conv_1",
            )
            conv2 = pt.layers.conv2d(
                conv1,
                32,
                kernel_size=[4, 4],
                strides=[2, 2],
                activation=pt.nn.elu,
                reuse=reuse,
                name="conv_2",
            )
            hidden = pt.layers.flatten(conv2)

        with pt.variable_scope(scope + "/" + "flat_encoding"):
            hidden_flat = ModelUtils.create_vector_observation_encoder(
                hidden, h_size, activation, num_layers, scope, reuse
            )
        return hidden_flat

    @staticmethod
    def create_nature_cnn_visual_observation_encoder(
        image_input: pt.Tensor,
        h_size: int,
        activation: ActivationFunction,
        num_layers: int,
        scope: str,
        reuse: bool,
    ) -> pt.Tensor:
        """
        Builds a set of resnet visual encoders.
        :param image_input: The placeholder for the image input to use.
        :param h_size: Hidden layer size.
        :param activation: What type of activation function to use for layers.
        :param num_layers: number of hidden layers to create.
        :param scope: The scope of the graph within which to create the ops.
        :param reuse: Whether to re-use the weights within the same scope.
        :return: List of hidden layer tensors.
        """
        with pt.variable_scope(scope):
            conv1 = pt.layers.conv2d(
                image_input,
                32,
                kernel_size=[8, 8],
                strides=[4, 4],
                activation=pt.nn.elu,
                reuse=reuse,
                name="conv_1",
            )
            conv2 = pt.layers.conv2d(
                conv1,
                64,
                kernel_size=[4, 4],
                strides=[2, 2],
                activation=pt.nn.elu,
                reuse=reuse,
                name="conv_2",
            )
            conv3 = pt.layers.conv2d(
                conv2,
                64,
                kernel_size=[3, 3],
                strides=[1, 1],
                activation=pt.nn.elu,
                reuse=reuse,
                name="conv_3",
            )
            hidden = pt.layers.flatten(conv3)

        with pt.variable_scope(scope + "/" + "flat_encoding"):
            hidden_flat = ModelUtils.create_vector_observation_encoder(
                hidden, h_size, activation, num_layers, scope, reuse
            )
        return hidden_flat

    @staticmethod
    def create_resnet_visual_observation_encoder(
        image_input: pt.Tensor,
        h_size: int,
        activation: ActivationFunction,
        num_layers: int,
        scope: str,
        reuse: bool,
    ) -> pt.Tensor:
        """
        Builds a set of resnet visual encoders.
        :param image_input: The placeholder for the image input to use.
        :param h_size: Hidden layer size.
        :param activation: What type of activation function to use for layers.
        :param num_layers: number of hidden layers to create.
        :param scope: The scope of the graph within which to create the ops.
        :param reuse: Whether to re-use the weights within the same scope.
        :return: List of hidden layer tensors.
        """
        n_channels = [16, 32, 32]  # channel for each stack
        n_blocks = 2  # number of residual blocks
        with pt.variable_scope(scope):
            hidden = image_input
            for i, ch in enumerate(n_channels):
                hidden = pt.layers.conv2d(
                    hidden,
                    ch,
                    kernel_size=[3, 3],
                    strides=[1, 1],
                    reuse=reuse,
                    name="layer%dconv_1" % i,
                )
                hidden = pt.layers.max_pooling2d(
                    hidden, pool_size=[3, 3], strides=[2, 2], padding="same"
                )
                # create residual blocks
                for j in range(n_blocks):
                    block_input = hidden
                    hidden = pt.nn.relu(hidden)
                    hidden = pt.layers.conv2d(
                        hidden,
                        ch,
                        kernel_size=[3, 3],
                        strides=[1, 1],
                        padding="same",
                        reuse=reuse,
                        name="layer%d_%d_conv1" % (i, j),
                    )
                    hidden = pt.nn.relu(hidden)
                    hidden = pt.layers.conv2d(
                        hidden,
                        ch,
                        kernel_size=[3, 3],
                        strides=[1, 1],
                        padding="same",
                        reuse=reuse,
                        name="layer%d_%d_conv2" % (i, j),
                    )
                    hidden = pt.add(block_input, hidden)
            hidden = pt.nn.relu(hidden)
            hidden = pt.layers.flatten(hidden)

        with pt.variable_scope(scope + "/" + "flat_encoding"):
            hidden_flat = ModelUtils.create_vector_observation_encoder(
                hidden, h_size, activation, num_layers, scope, reuse
            )
        return hidden_flat

    @staticmethod
    def get_encoder_for_type(encoder_type: EncoderType) -> EncoderFunction:
        ENCODER_FUNCTION_BY_TYPE = {
            EncoderType.SIMPLE: ModelUtils.create_visual_observation_encoder,
            EncoderType.NATURE_CNN: ModelUtils.create_nature_cnn_visual_observation_encoder,
            EncoderType.RESNET: ModelUtils.create_resnet_visual_observation_encoder,
        }
        return ENCODER_FUNCTION_BY_TYPE.get(
            encoder_type, ModelUtils.create_visual_observation_encoder
        )

    @staticmethod
    def break_into_branches(
        concatenated_logits: pt.Tensor, action_size: List[int]
    ) -> List[pt.Tensor]:
        """
        Takes a concatenated set of logits that represent multiple discrete action branches
        and breaks it up into one Tensor per branch.
        :param concatenated_logits: Tensor that represents the concatenated action branches
        :param action_size: List of ints containing the number of possible actions for each branch.
        :return: A List of Tensors containing one tensor per branch.
        """
        action_idx = [0] + list(np.cumsum(action_size))
        branched_logits = [
            concatenated_logits[:, action_idx[i] : action_idx[i + 1]]
            for i in range(len(action_size))
        ]
        return branched_logits

    @staticmethod
    def create_discrete_action_masking_layer(
        branches_logits: List[pt.Tensor],
        action_masks: pt.Tensor,
        action_size: List[int],
    ) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        """
        Creates a masking layer for the discrete actions
        :param branches_logits: A List of the unnormalized action probabilities for each branch
        :param action_masks: The mask for the logits. Must be of dimension [None x total_number_of_action]
        :param action_size: A list containing the number of possible actions for each branch
        :return: The action output dimension [batch_size, num_branches], the concatenated
            normalized probs (after softmax)
        and the concatenated normalized log probs
        """
        branch_masks = ModelUtils.break_into_branches(action_masks, action_size)
        raw_probs = [
            pt.multiply(pt.nn.softmax(branches_logits[k]) + EPSILON, branch_masks[k])
            for k in range(len(action_size))
        ]
        normalized_probs = [
            pt.divide(raw_probs[k], pt.reduce_sum(raw_probs[k], axis=1, keepdims=True))
            for k in range(len(action_size))
        ]
        output = pt.concat(
            [
                pt.multinomial(pt.log(normalized_probs[k] + EPSILON), 1)
                for k in range(len(action_size))
            ],
            axis=1,
        )
        return (
            output,
            pt.concat([normalized_probs[k] for k in range(len(action_size))], axis=1),
            pt.concat(
                [
                    pt.log(normalized_probs[k] + EPSILON)
                    for k in range(len(action_size))
                ],
                axis=1,
            ),
        )

    @staticmethod
    def _check_resolution_for_encoder(
        vis_in: pt.Tensor, vis_encoder_type: EncoderType
    ) -> None:
        min_res = ModelUtils.MIN_RESOLUTION_FOR_ENCODER[vis_encoder_type]
        height = vis_in.shape[1]
        width = vis_in.shape[2]
        if height < min_res or width < min_res:
            raise UnityTrainerException(
                f"Visual observation resolution ({width}x{height}) is too small for"
                f"the provided EncoderType ({vis_encoder_type.value}). The min dimension is {min_res}"
            )

    @staticmethod
    def create_observation_streams(
        visual_in: List[pt.Tensor],
        vector_in: pt.Tensor,
        num_streams: int,
        h_size: int,
        num_layers: int,
        vis_encode_type: EncoderType = EncoderType.SIMPLE,
        stream_scopes: List[str] = None,
    ) -> List[pt.Tensor]:
        """
        Creates encoding stream for observations.
        :param num_streams: Number of streams to create.
        :param h_size: Size of hidden linear layers in stream.
        :param num_layers: Number of hidden linear layers in stream.
        :param stream_scopes: List of strings (length == num_streams), which contains
            the scopes for each of the streams. None if all under the same PT scope.
        :return: List of encoded streams.
        """
        activation_fn = ModelUtils.swish
        vector_observation_input = vector_in

        final_hiddens = []
        for i in range(num_streams):
            # Pick the encoder function based on the EncoderType
            create_encoder_func = ModelUtils.get_encoder_for_type(vis_encode_type)

            visual_encoders = []
            hidden_state, hidden_visual = None, None
            _scope_add = stream_scopes[i] if stream_scopes else ""
            if len(visual_in) > 0:
                for j, vis_in in enumerate(visual_in):
                    ModelUtils._check_resolution_for_encoder(vis_in, vis_encode_type)
                    encoded_visual = create_encoder_func(
                        vis_in,
                        h_size,
                        activation_fn,
                        num_layers,
                        f"{_scope_add}main_graph_{i}_encoder{j}",  # scope
                        False,  # reuse
                    )
                    visual_encoders.append(encoded_visual)
                hidden_visual = pt.concat(visual_encoders, axis=1)
            if vector_in.get_shape()[-1] > 0:
                # Don't encode non-existant or 0-shape inputs
                hidden_state = ModelUtils.create_vector_observation_encoder(
                    vector_observation_input,
                    h_size,
                    activation_fn,
                    num_layers,
                    scope=f"{_scope_add}main_graph_{i}",
                    reuse=False,
                )
            if hidden_state is not None and hidden_visual is not None:
                final_hidden = pt.concat([hidden_visual, hidden_state], axis=1)
            elif hidden_state is None and hidden_visual is not None:
                final_hidden = hidden_visual
            elif hidden_state is not None and hidden_visual is None:
                final_hidden = hidden_state
            else:
                raise Exception(
                    "No valid network configuration possible. "
                    "There are no states or observations in this brain"
                )
            final_hiddens.append(final_hidden)
        return final_hiddens

    @staticmethod
    def create_recurrent_encoder(input_state, memory_in, sequence_length, name="lstm"):
        """
        Builds a recurrent encoder for either state or observations (LSTM).
        :param sequence_length: Length of sequence to unroll.
        :param input_state: The input tensor to the LSTM cell.
        :param memory_in: The input memory to the LSTM cell.
        :param name: The scope of the LSTM cell.
        """
        s_size = input_state.get_shape().as_list()[1]
        m_size = memory_in.get_shape().as_list()[1]
        lstm_input_state = pt.reshape(input_state, shape=[-1, sequence_length, s_size])
        memory_in = pt.reshape(memory_in[:, :], [-1, m_size])
        half_point = int(m_size / 2)
        with pt.variable_scope(name):
            rnn_cell = pt.nn.rnn_cell.BasicLSTMCell(half_point)
            lstm_vector_in = pt.nn.rnn_cell.LSTMStateTuple(
                memory_in[:, :half_point], memory_in[:, half_point:]
            )
            recurrent_output, lstm_state_out = pt.nn.dynamic_rnn(
                rnn_cell, lstm_input_state, initial_state=lstm_vector_in
            )

        recurrent_output = pt.reshape(recurrent_output, shape=[-1, half_point])
        return recurrent_output, pt.concat([lstm_state_out.c, lstm_state_out.h], axis=1)

    @staticmethod
    def create_value_heads(
        stream_names: List[str], hidden_input: pt.Tensor
    ) -> Tuple[Dict[str, pt.Tensor], pt.Tensor]:
        """
        Creates one value estimator head for each reward signal in stream_names.
        Also creates the node corresponding to the mean of all the value heads in self.value.
        self.value_head is a dictionary of stream name to node containing the value estimator head for that signal.
        :param stream_names: The list of reward signal names
        :param hidden_input: The last layer of the Critic. The heads will consist of one dense hidden layer on top
        of the hidden input.
        """
        value_heads = {}
        for name in stream_names:
            value = pt.layers.dense(hidden_input, 1, name="{}_value".format(name))
            value_heads[name] = value
        value = pt.reduce_mean(list(value_heads.values()), 0)
        return value_heads, value
