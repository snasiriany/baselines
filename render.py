from gym_recording import playback

i = 0
def handle_ep(observations, actions, rewards):
    global i
    print(i, observations.shape, actions.shape, len(rewards))
    i = i + 1
    pass

playback.scan_recorded_traces("/tmp/openai.gym.1506013653.60697.12439", handle_ep)
