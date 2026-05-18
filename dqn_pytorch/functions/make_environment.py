import gymnasium


def make_environment(environment_name: str, seed: int):
    def thunk():
        env = gymnasium.make(environment_name)
        env = gymnasium.wrappers.AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=True,
            grayscale_obs=True,
            grayscale_newaxis=False,
        )
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = gymnasium.wrappers.ClipReward(env, min_reward=-1.0, max_reward=1.0)
        env = gymnasium.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)
        return env

    return thunk
