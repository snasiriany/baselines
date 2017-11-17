#!/usr/bin/env python
# noinspection PyUnresolvedReferences
import mujoco_py # Mujoco must come before other imports. https://openai.slack.com/archives/C1H6P3R7B/p1492828680631850
from mpi4py import MPI
from baselines.common import set_global_seeds
import os.path as osp
import gym
import logging
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.common.mpi_fork import mpi_fork
from baselines import bench
from baselines.trpo_mpi import trpo_mpi
import sys

import os
from time import strftime, localtime
from gym.envs.registration import register

from gym_recording.wrappers import TraceRecordingWrapper #record datat

def train(env_id, num_timesteps, seed):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    register(
        id='HalfCheetah-v3',
        entry_point='R.rl.demo.env:HalfCheetahTrpoEnv',
    )

    env = gym.make(env_id)
    #env = TraceRecordingWrapper(env) #track the video
    #print(env.directory)

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
            hid_size=32, num_hid_layers=2)

    basedir = "/data/soroush/experiments/trpo/" + env_id
    logdir = os.path.join(basedir, strftime("%Y-%m-%d|%H:%M:%S", localtime()))
    logger.configure(dir=logdir)
    print("logging directory:", logger.get_dir())
    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), "%i.monitor.json" % rank), allow_early_resets=True)
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    video_freq = 10
    timesteps_per_batch = 2048

    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=timesteps_per_batch, max_kl=0.01, cg_iters=10, cg_damping=0.1,
        max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3, video_freq=video_freq)
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='HalfCheetah-v3')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    args = parser.parse_args()
    train(args.env, num_timesteps=2e6, seed=args.seed)


if __name__ == '__main__':
    main()
