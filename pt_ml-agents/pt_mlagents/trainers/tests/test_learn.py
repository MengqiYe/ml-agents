import pytest
import yaml
from unittest.mock import MagicMock, patch, mock_open
from pt_mlagents.trainers import learn
from pt_mlagents.trainers.trainer_controller import TrainerController
from pt_mlagents.trainers.learn import parse_command_line
from pt_mlagents.trainers.cli_utils import DetectDefault
from pt_mlagents_envs.exception import UnityEnvironmentException
from pt_mlagents.trainers.stats import StatsReporter
from pt_mlagents.trainers.environment_parameter_manager import EnvironmentParameterManager


def basic_options(extra_args=None):
    extra_args = extra_args or {}
    args = ["basic_path"]
    if extra_args:
        args += [f"{k}={v}" for k, v in extra_args.items()]
    return parse_command_line(args)


MOCK_YAML = """
    behaviors:
        {}
    """

MOCK_INITIALIZE_YAML = """
    behaviors:
        {}
    checkpoint_settings:
        initialize_from: notuselessrun
    """

MOCK_PARAMETER_YAML = """
    behaviors:
        {}
    env_settings:
        env_path: "./oldenvfile"
        num_envs: 4
        base_port: 4001
        seed: 9870
    checkpoint_settings:
        run_id: uselessrun
        initialize_from: notuselessrun
    debug: false
    """


@patch("pt_mlagents.trainers.learn.write_timing_tree")
@patch("pt_mlagents.trainers.learn.write_run_options")
@patch("pt_mlagents.trainers.learn.handle_existing_directories")
@patch("pt_mlagents.trainers.learn.TrainerFactory")
@patch("pt_mlagents.trainers.learn.SubprocessEnvManager")
@patch("pt_mlagents.trainers.learn.create_environment_factory")
@patch("pt_mlagents.trainers.settings.load_config")
def test_run_training(
    load_config,
    create_environment_factory,
    subproc_env_mock,
    trainer_factory_mock,
    handle_dir_mock,
    write_run_options_mock,
    write_timing_tree_mock,
):
    mock_env = MagicMock()
    mock_env.external_brain_names = []
    mock_env.academy_name = "TestAcademyName"
    create_environment_factory.return_value = mock_env
    load_config.return_value = yaml.safe_load(MOCK_INITIALIZE_YAML)
    mock_param_manager = MagicMock(return_value="mock_param_manager")
    mock_init = MagicMock(return_value=None)
    with patch.object(EnvironmentParameterManager, "__new__", mock_param_manager):
        with patch.object(TrainerController, "__init__", mock_init):
            with patch.object(TrainerController, "start_learning", MagicMock()):
                options = basic_options()
                learn.run_training(0, options)
                mock_init.assert_called_once_with(
                    trainer_factory_mock.return_value,
                    "results/ppo",
                    "ppo",
                    "mock_param_manager",
                    True,
                    0,
                )
                handle_dir_mock.assert_called_once_with(
                    "results/ppo", False, False, "results/notuselessrun"
                )
                write_timing_tree_mock.assert_called_once_with("results/ppo/run_logs")
                write_run_options_mock.assert_called_once_with("results/ppo", options)
    StatsReporter.writers.clear()  # make sure there aren't any writers as added by learn.py


def test_bad_env_path():
    with pytest.raises(UnityEnvironmentException):
        factory = learn.create_environment_factory(
            env_path="/foo/bar",
            no_graphics=True,
            seed=-1,
            start_port=8000,
            env_args=None,
            log_folder="results/log_folder",
        )
        factory(worker_id=-1, side_channels=[])


@patch("builtins.open", new_callable=mock_open, read_data=MOCK_YAML)
def test_commandline_args(mock_file):
    # No args raises
    # with pytest.raises(SystemExit):
    #     parse_command_line([])
    # Test with defaults
    opt = parse_command_line(["mytrainerpath"])
    assert otorch.behaviors == {}
    assert otorch.env_settings.env_path is None
    assert otorch.checkpoint_settings.resume is False
    assert otorch.checkpoint_settings.inference is False
    assert otorch.checkpoint_settings.run_id == "ppo"
    assert otorch.checkpoint_settings.initialize_from is None
    assert otorch.env_settings.seed == -1
    assert otorch.env_settings.base_port == 5005
    assert otorch.env_settings.num_envs == 1
    assert otorch.engine_settings.no_graphics is False
    assert otorch.debug is False
    assert otorch.env_settings.env_args is None

    full_args = [
        "mytrainerpath",
        "--env=./myenvfile",
        "--resume",
        "--inference",
        "--run-id=myawesomerun",
        "--seed=7890",
        "--train",
        "--base-port=4004",
        "--initialize-from=testdir",
        "--num-envs=2",
        "--no-graphics",
        "--debug",
    ]

    opt = parse_command_line(full_args)
    assert otorch.behaviors == {}
    assert otorch.env_settings.env_path == "./myenvfile"
    assert otorch.checkpoint_settings.run_id == "myawesomerun"
    assert otorch.checkpoint_settings.initialize_from == "testdir"
    assert otorch.env_settings.seed == 7890
    assert otorch.env_settings.base_port == 4004
    assert otorch.env_settings.num_envs == 2
    assert otorch.engine_settings.no_graphics is True
    assert otorch.debug is True
    assert otorch.checkpoint_settings.inference is True
    assert otorch.checkpoint_settings.resume is True


@patch("builtins.open", new_callable=mock_open, read_data=MOCK_PARAMETER_YAML)
def test_yaml_args(mock_file):
    # Test with opts loaded from YAML
    DetectDefault.non_default_args.clear()
    opt = parse_command_line(["mytrainerpath"])
    assert otorch.behaviors == {}
    assert otorch.env_settings.env_path == "./oldenvfile"
    assert otorch.checkpoint_settings.run_id == "uselessrun"
    assert otorch.checkpoint_settings.initialize_from == "notuselessrun"
    assert otorch.env_settings.seed == 9870
    assert otorch.env_settings.base_port == 4001
    assert otorch.env_settings.num_envs == 4
    assert otorch.engine_settings.no_graphics is False
    assert otorch.debug is False
    assert otorch.env_settings.env_args is None
    # Test that CLI overrides YAML
    full_args = [
        "mytrainerpath",
        "--env=./myenvfile",
        "--resume",
        "--inference",
        "--run-id=myawesomerun",
        "--seed=7890",
        "--train",
        "--base-port=4004",
        "--num-envs=2",
        "--no-graphics",
        "--debug",
    ]

    opt = parse_command_line(full_args)
    assert otorch.behaviors == {}
    assert otorch.env_settings.env_path == "./myenvfile"
    assert otorch.checkpoint_settings.run_id == "myawesomerun"
    assert otorch.env_settings.seed == 7890
    assert otorch.env_settings.base_port == 4004
    assert otorch.env_settings.num_envs == 2
    assert otorch.engine_settings.no_graphics is True
    assert otorch.debug is True
    assert otorch.checkpoint_settings.inference is True
    assert otorch.checkpoint_settings.resume is True


@patch("builtins.open", new_callable=mock_open, read_data=MOCK_YAML)
def test_env_args(mock_file):
    full_args = [
        "mytrainerpath",
        "--env=./myenvfile",
        "--env-args",  # Everything after here will be grouped in a list
        "--foo=bar",
        "--blah",
        "baz",
        "100",
    ]

    opt = parse_command_line(full_args)
    assert otorch.env_settings.env_args == ["--foo=bar", "--blah", "baz", "100"]
