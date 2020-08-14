from abc import ABC, abstractmethod

from pt_mlagents_envs.base_env import DecisionSteps
from pt_mlagents.trainers.action_info import ActionInfo


class Policy(ABC):
    @abstractmethod
    def get_action(
        self, decision_requests: DecisionSteps, worker_id: int = 0
    ) -> ActionInfo:
        pass
