from diffusion_policy.env_runner.base_runner import BaseLowdimRunner


class NullRunner(BaseLowdimRunner):
    """No-op runner for offline training where no simulation environment is needed."""

    def __init__(self, output_dir):
        super().__init__(output_dir)

    def run(self, policy):
        return {}
