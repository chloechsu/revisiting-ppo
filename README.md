# Revisiting Design Choices in Proximal Policy Optimization

The PPO implementation is based on the open-source code for ICLR 2020 paper
"Implementation Matters in Deep RL: A Case Study on PPO and TRPO":
<https://github.com/implementation-matters/code-for-paper>.

All our analysis and plots are produced via Jupyter notebooks in the ``analysis`` folder.

## Failure mode examples

The failure mode example environments are defined in ``analysis/envs.py``, and the experiments are analyzed in the corresponding Jupyter notebooks in the ``analysis`` folder.

## Beta policy implementation

The Beta policy is implemented as a ``CtsBetaPolicy`` class in ``src/policy_gradients/models.py``.

## MuJoCo experiments

We assume that the user has a machine with MuJoCo and mujoco\_py properly set up and installed. To see if MuJoCo is properly installed, try running the following:

```python
import gym
gym.make_env("Humanoid-v2")
```

The dependencies are listed in the ``src/requirements.txt`` file, can be installed via ``pip install -r requirements.txt``.

As an example, to reproduce our MuJoCo Gaussian vs beta policy comparison figures: run the following commands:
1. ``cd src/gaussian_vs_beta/``
2. ``python setup_agents.py``: the setup\_agents.py script contains detailed
experiments settings and sets up configuration files for each agent.
3. ``cd ../``
4. Edit the ``NUM_THREADS`` variables in the ``run_agents.py`` file according to your local machine.
5. Train the agents: ``python run_agents.py gaussian_vs_beta/agent_configs``
6. Plot results in the corresponding Jupyter notebook in the analysis folder.

For other MuJoCo comparisons, similarly see the agent setup files in  ``src/kl_direction`` and ``src/base_exp``, or create your own custom agent setup file with the desired configurations.

For more details about the code, see the README file in the original GitHub repo:
<https://github.com/implementation-matters/code-for-paper>.


