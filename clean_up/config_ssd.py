from clean_up.utils.configdict import ConfigDict

def get_config():

    config = ConfigDict()

    config.env = ConfigDict()
    config.env.allow_giving = False
    config.env.asymmetric = False
    config.env.beam_width = 1
    config.env.cleaning_penalty = 0
    config.env.disable_left_right_action = False
    config.env.disable_rotation_action = True
    # if not None, a fixed global reference frame is used for all agents
    config.env.global_ref_point = None
    config.env.map_name = 'cleanup_10x10_sym'
    config.env.max_steps = 50
    config.env.n_agents = 4
    config.env.name = 'ssd'
    config.env.obs_cleaned_1hot = False
    config.env.obs_height = 5
    config.env.obs_width = 5
    config.env.reward_value = 1.0
    config.env.random_orientation = False
    config.env.shuffle_spawn = False
    config.env.view_size = 2  # 0.5(height - 1)
    config.env.cleanup_params = ConfigDict()
    config.env.cleanup_params.appleRespawnProbability = 0.3  # 10x10 0.3 | small 0.5 | large 0.125      default=0.125
    config.env.cleanup_params.thresholdDepletion = 0.4  # 10x10 0.4 | small 0.6             default=0.4
    config.env.cleanup_params.thresholdRestoration = 0.0  # 10x10 0.0 | small 0             default=0.0
    config.env.cleanup_params.wasteSpawnProbability = 0.5  # 10x10 0.5 | small 0.5          default=0.5

    return config
