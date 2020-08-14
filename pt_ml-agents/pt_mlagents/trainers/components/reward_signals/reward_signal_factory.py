from typing import Dict, Type
from pt_mlagents.trainers.exception import UnityTrainerException
from pt_mlagents.trainers.components.reward_signals import RewardSignal
from pt_mlagents.trainers.components.reward_signals.extrinsic.signal import (
    ExtrinsicRewardSignal,
)
from pt_mlagents.trainers.components.reward_signals.gail.signal import GAILRewardSignal
from pt_mlagents.trainers.components.reward_signals.curiosity.signal import (
    CuriosityRewardSignal,
)
from pt_mlagents.trainers.policy.tf_policy import TFPolicy
from pt_mlagents.trainers.settings import RewardSignalSettings, RewardSignalType


NAME_TO_CLASS: Dict[RewardSignalType, Type[RewardSignal]] = {
    RewardSignalType.EXTRINSIC: ExtrinsicRewardSignal,
    RewardSignalType.CURIOSITY: CuriosityRewardSignal,
    RewardSignalType.GAIL: GAILRewardSignal,
}


def create_reward_signal(
    policy: TFPolicy, name: RewardSignalType, settings: RewardSignalSettings
) -> RewardSignal:
    """
    Creates a reward signal class based on the name and config entry provided as a dict.
    :param policy: The policy class which the reward will be applied to.
    :param name: The name of the reward signal
    :param config_entry: The config entries for that reward signal
    :return: The reward signal class instantiated
    """
    rcls = NAME_TO_CLASS.get(name)
    if not rcls:
        raise UnityTrainerException("Unknown reward signal type {0}".format(name))

    class_inst = rcls(policy, settings)
    return class_inst
