import matplotlib.pyplot as plt
import numpy as np

import json
from pprint import pprint
import os

exps = [
    ("mbmf", "HalfCheetah-v3", "2017-11-15|21:19:19"),
    ("mbmf", "HalfCheetah-v3", "2017-11-07|23:55:27"),
    ("mbmf", "HalfCheetah-v3", "2017-11-07|23:50:32"),
    ("mbmf", "HalfCheetah-v3", "2017-11-06|22:35:14"),
    ("mbmf", "HalfCheetah-v3", "2017-11-01|23:32:33"),
    ("mbmf", "HalfCheetah-v3", "2017-10-31|18:26:44"),
    ("mbmf", "HalfCheetah-v3", "2017-10-31|17:39:09"),
    ("mbmf", "HalfCheetah-v3", "2017-10-31|16:04:13"),
    ("mbmf", "HalfCheetah-v3", "2017-10-31|15:38:48"),
    ("mbmf", "HalfCheetah-v3", "2017-10-31|15:44:39"),
    ("trpo", "HalfCheetah-v3", "2017-10-31|14:37:00"),
    ("trpo", "HalfCheetah-v3", "2017-10-31|12:53:45"),
    ("trpo", "HalfCheetah-v3", "2017-10-31|12:44:11"),
    ("mbmf", "HalfCheetah-v3", "2017-10-31|13:23:24"),
    ("mbmf", "HalfCheetah-v3", "2017-10-31|13:20:43"),
    ("trpo", "HalfCheetah-v1", "2017-10-24|18:40:40"),
    ("trpo", "HalfCheetah-v1", "2017-10-24|18:06:10"),
    ("mbmf", "HalfCheetah-v1", "2017-10-21|21:04:02"),
    ("trpo", "HalfCheetah-v1", "2017-10-24|20:41:06"),
    ("mbmf", "HalfCheetah-v1", "2017-10-21|20:24:17"),
    ("mbmf", "HalfCheetah-v1", "2017-10-21|19:28:06"),
    ("trpo", "Pusher-v0", "2017-10-25|14:00:55"),
    ("trpo", "Pusher-v0", "2017-10-25|16:32:24"),
    ("trpo", "Pusher-v0", "2017-10-27|17:19:53"),
    ("mbmf", "HalfCheetah-v2", "2017-10-29|20:09:38"),
    ("mbmf", "HalfCheetah-v3", "2017-10-29|20:34:13"),
    ("mbmf", "HalfCheetah-v3", "2017-10-30|23:36:52"),
]

for (alg_type, env_type, exp_time) in exps:
    base_dir = os.path.join("/data/soroush/experiments/",
            alg_type, env_type, exp_time)

    iterations = []

    with open(os.path.join(base_dir, "progress.json"), 'rt') as fh:
        lines = fh.readlines()

    EpLenMean = []
    EpRewMean = []
    EpThisIter = []
    EpisodesSoFar = []
    TimeElapsed = []
    TimestepsSoFar = []
    entloss = []
    entropy = []
    ev_tdlam_before = []
    meankl = []
    optimgain = []
    surrgain = []

    for line in lines:
        iteration = json.loads(line)
        iterations.append(iteration)
        EpLenMean.append(iteration["EpLenMean"])
        EpRewMean.append(iteration["EpRewMean"])
        EpThisIter.append(iteration["EpThisIter"])
        EpisodesSoFar.append(iteration["EpisodesSoFar"])
        TimeElapsed.append(iteration["TimeElapsed"])
        TimestepsSoFar.append(iteration["TimestepsSoFar"])
        entloss.append(iteration["entloss"])
        entropy.append(iteration["entropy"])
        ev_tdlam_before.append(iteration["ev_tdlam_before"])
        meankl.append(iteration["meankl"])
        optimgain.append(iteration["optimgain"])
        surrgain.append(iteration["surrgain"])

    plt.scatter(np.arange(0, len(iterations)), EpRewMean, s=1)
    plt.ylabel('EpRewMean')
    plt.xlabel('iteration')
    plt.savefig(os.path.join(base_dir, "EpRewMean.png"))
    plt.close()

    plt.scatter(np.arange(0, len(iterations)), optimgain, s=1)
    plt.ylabel('optimgain')
    plt.xlabel('iteration')
    plt.savefig(os.path.join(base_dir, "optimgain.png"))
    plt.close()

    plt.scatter(np.arange(0, len(iterations)), surrgain, s=1)
    plt.ylabel('surrgain')
    plt.xlabel('iteration')
    plt.savefig(os.path.join(base_dir, "surrgain.png"))
    plt.close()

    plt.scatter(np.arange(0, len(iterations)), meankl, s=1)
    plt.ylabel('meankl')
    plt.xlabel('iteration')
    plt.savefig(os.path.join(base_dir, "meankl.png"))
    plt.close()

    plt.scatter(np.arange(0, len(iterations)), entropy, s=1)
    plt.ylabel('entropy')
    plt.xlabel('iteration')
    plt.savefig(os.path.join(base_dir, "entropy.png"))
    plt.close()
