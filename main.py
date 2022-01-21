from agent import RL_Agents
import time
import pandas as pd
from citylearn_3dem import CityLearn_3dem
from pathlib import Path
from citylearn_3dem import RBC_Agent
import matplotlib.pyplot as plt
from stable_baselines import SAC
from stable_baselines.sac.policies import FeedForwardPolicy
from functions import economic_cost, discomfort

class CustomSACPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                              layers=[256] * 2,
                                              layer_norm=True,
                                              feature_extraction="mlp")


# Central agent controlling one of the buildings using the OpenAI Stable Baselines
climate_zone = 1
data_path = Path("data/Climate_Zone_"+str(climate_zone))
building_attributes = data_path / 'building_attributes.json'
weather_file = data_path / 'weather_data.csv'
solar_profile = data_path / 'solar_generation_1kW.csv'
building_state_actions = Path("buildings_state_action_space.json")
building_dynamics_state = Path("buildings_dynamics_state_space.json")
el_data = data_path / 'electricity_price.csv'
building_ids = ['Building_1']

objective_function = ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption','quadratic']

env = CityLearn_3dem(data_path, building_attributes, weather_file, solar_profile, el_data, building_ids,
                     buildings_states_actions = building_state_actions, building_dynamics_state= building_dynamics_state, simulation_period=(3624+12,5831),  # initial timestep + max(lookback) because the first lookback periods are taken as inputs for the neural network
                     cost_function = objective_function, central_agent = False, verbose = 1)


observations_spaces, actions_spaces = env.get_state_action_spaces()
env.reset()
done = False


#------------------------No storage case with fixed demand------------------------------
while not done:

    _, rewards, done, _ = env.step([[0 for _ in range(len(actions_spaces[i].sample()))] for i in range(len(building_ids))])

cost_no_es = env.cost()


cost_no_storage = cost_no_es # costs from CityLearn_3dem with no Energy Storage

#------------------------RBC------------------------------

'''IMPORTANT: Make sure that the buildings_state_action_space.json file contains the hour of day as 3rd true state:
{"Building_1": {
    "states": {
        "month": true,
        "day": true,
        "hour": true
Alternative, modify the line: "hour_day = states[0][2]" of the RBC_Agent Class in agent.py
'''
env.reset()
env = CityLearn_3dem(data_path, building_attributes, weather_file, solar_profile, el_data, building_ids,
                     buildings_states_actions = building_state_actions, building_dynamics_state= building_dynamics_state, simulation_period=(3624+12,5831),
                     cost_function = objective_function, central_agent = False, verbose = 1)


# Instantiating the control agent(s)
agents = RBC_Agent(actions_spaces)

state = env.reset()
done = False
rewards_list = []
start = time.time()
azionirbc=[]
while not done:
    action = agents.select_action(state)
    next_state, rewards, done, _ = env.step(action)
    state = next_state
    rewards_list.append(rewards)
cost_rbc = env.cost()
end = time.time()
print(end-start)
cost_rbc


#------------------------SAC-stable_baseline ------------------------------
env.reset()
env = CityLearn_3dem(data_path, building_attributes, weather_file, solar_profile, el_data, building_ids,
                     buildings_states_actions = building_state_actions, building_dynamics_state= building_dynamics_state, simulation_period=(3624+12,5831),
                     cost_function = objective_function, central_agent = True, verbose = 1)

#Reinforcement learning initialization model

model = SAC(CustomSACPolicy, env, verbose=0, learning_rate=0.001,
            gamma=0.9, tau=0.005, batch_size=512,ent_coef=0.1, learning_starts=2195, target_entropy='auto')
start = time.time()
model.learn(total_timesteps=2195*2,log_interval=1000)
end = time.time()
print(end-start)

env = CityLearn_3dem(data_path, building_attributes, weather_file, solar_profile, el_data, building_ids,
                     buildings_states_actions = building_state_actions, building_dynamics_state= building_dynamics_state, simulation_period=(3624+12,5831),
                     cost_function = objective_function, central_agent = True, verbose = 1)


obs = env.reset()
dones = False
counter = []
actions = []
while dones==False:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    counter.append(rewards)
    actions.append(action)
cost_sac = env.cost()



Tlstm = env.buildings['Building_1'].lstm_results['T_lstm']
Set_point = env.buildings['Building_1'].sim_results['set_point']

plt.plot(Tlstm[:200])
plt.plot(Set_point[:200])
plt.show()



costo_sac,costo_tot = economic_cost(env,building_ids)
discomfort = discomfort(env,building_ids)


