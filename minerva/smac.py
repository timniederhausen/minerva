import os
import time
import logging
import typing

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.optimizer.smbo import SMBO
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger

log = logging.getLogger(__name__)


class ScenarioProperties:
    r"""This type just exists to provide correct type hints for :class:`.Scenario` objects.

    Internally, Scenario.__init__ does `setattr(*kw) for kw in kwargs.items()`, making IDE support impossible.
    """
    run_obj: str
    cs: ConfigurationSpace
    deterministic: bool
    output_dir: str


class ScenarioWithSavepoint(Scenario):
    r"""Custom :class:`.Scenario` with support for an arbitrary callback upon save.

    This allows the user to back SMAC's state up, as soon as any meaningful progress has been made.
    """

    def __init__(self, scenario=None, cmd_options: dict = None,
                 save_callback: typing.Callable[[], None] = None):
        super().__init__(scenario, cmd_options)
        self.save_callback = save_callback

    def save(self):
        if self.save_callback:
            self.save_callback()


class IncrementalSMBO(SMBO):
    r"""Incremental SMBO optimizer

    The main difference between this class and its base is that after each interation in :func:`.run`,
    the optimizer's state is saved to disk. This ensures that progress isn't lost.
    """

    def run(self):
        self.start()

        # Main BO loop
        while True:
            start_time = time.time()
            X, Y = self.rh2EPM.transform(self.runhistory)

            self.logger.debug("Search for next configuration")
            # get all found configurations sorted according to acq
            challengers = self.choose_next(X, Y)

            time_spent = time.time() - start_time
            time_left = self._get_timebound_for_intensification(time_spent)

            self.logger.debug("Intensify")

            self.incumbent, inc_perf = self.intensifier.intensify(
                challengers=challengers,
                incumbent=self.incumbent,
                run_history=self.runhistory,
                aggregate_func=self.aggregate_func,
                time_bound=max(self.intensifier._min_time, time_left))

            self.runhistory.save_json(fn=os.path.join(self.scenario.output_dir_for_this_run, 'runhistory.json'))
            self.stats.save()
            self.scenario.save()

            logging.debug("Remaining budget: %f (wallclock), %f (ta costs), %f (target runs)" % (
                self.stats.get_remaing_time_budget(),
                self.stats.get_remaining_ta_budget(),
                self.stats.get_remaining_ta_runs()))

            if self.stats.is_budget_exhausted():
                break

            self.stats.print_stats(debug_out=True)

        return self.incumbent


def restore_state(scenario: typing.Union[Scenario, ScenarioProperties]):
    r"""Read in files for state-restoration: runhistory, stats, trajectory.

    :param scenario: Scenario whose state shall be loaded.
    :return: (RunHistory, Stats, dict)-tuple
    """
    # Check for folder and files
    rh_path = os.path.join(scenario.output_dir_for_this_run, 'runhistory.json')
    stats_path = os.path.join(scenario.output_dir_for_this_run, 'stats.json')
    traj_path_aclib = os.path.join(scenario.output_dir_for_this_run, 'traj_aclib2.json')
    if not os.path.isdir(scenario.output_dir_for_this_run):
        raise FileNotFoundError('Could not find folder from which to restore.')

    # Load runhistory and stats
    rh = RunHistory(aggregate_func=None)
    rh.load_json(rh_path, scenario.cs)
    log.debug('Restored runhistory from %s', rh_path)

    stats = Stats(scenario)
    stats.load(stats_path)
    log.debug('Restored stats from %s', stats_path)

    trajectory = TrajLogger.read_traj_aclib_format(fn=traj_path_aclib, cs=scenario.cs)
    incumbent = trajectory[-1]['incumbent']
    log.debug('Restored incumbent %s from %s', incumbent, traj_path_aclib)
    return rh, stats, incumbent


class ResumableSMAC:
    r"""Resumable SMAC facade for hyperparameter optimization.

    This is the main class to use when resumable behavior is desired. Optionally, a :class:`ScenarioWithSavepoint`
    can be used to perform custom actions once state has been saved.
    """

    def __init__(self, scenario: typing.Union[ScenarioWithSavepoint, Scenario, ScenarioProperties], seed=1):
        self.scenario = scenario
        self.scenario.output_dir_for_this_run = os.path.join(self.scenario.output_dir, 'run_1')

        # Create defaults
        rh = None
        stats = None
        incumbent = None

        if os.path.exists(self.scenario.output_dir_for_this_run):
            rh, stats, incumbent = restore_state(self.scenario)

        self.smac = SMAC4HPO(scenario=self.scenario,
                             rng=np.random.RandomState(seed),
                             runhistory=rh,
                             initial_design_kwargs=dict(n_configs_x_params=1),
                             stats=stats,
                             restore_incumbent=incumbent,
                             run_id=1,
                             smbo_class=IncrementalSMBO,
                             tae_runner=self)

    def __call__(self, cfg):
        raise NotImplemented()
