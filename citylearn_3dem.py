import gym
from gym.utils import seeding
import numpy as np
import pandas as pd
import json
from gym import spaces
from energy_models import HeatPump, ElectricHeater, EnergyStorage, Building, BuildingDynamics, cop_T, cop_curve_cooling
from reward_function import reward_function_sa, reward_function_ma
import torch
from pathlib import Path

#load max and min csv to normalize inputs of lstm
path = Path("data/Climate_Zone_"+str(1)) #1 -->climate zone
max_data_file = 'max.csv'
max_data = path / max_data_file
safety_factor = 1.15
with open(max_data) as csv_file:
    max_value = pd.read_csv(csv_file)
min_data_file = 'min.csv'
min_data = path / min_data_file
with open(min_data) as csv_file:
    min_value = pd.read_csv(csv_file)

# Reference Rule-based controller. Used as a baseline to calculate the costs in CityLearn_3dem
# It requires, at least, the hour of the day as input state
class RBC_Agent:
    def __init__(self, actions_spaces):
        self.actions_spaces = actions_spaces
        self.reset_action_tracker()

    def reset_action_tracker(self):
        self.action_tracker = []

    def select_action(self, states):
        hour_day = states[0][2]

        # Daytime: release stored energy
        a = [[0.0 for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        if hour_day >= 8 and hour_day <= 21:
            a = [[-0.08 for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]

        # Early nightime: store DHW and/or cooling energy
        if (hour_day >= 1 and hour_day <= 7) or (hour_day >= 22 and hour_day <= 24):
            a = []
            for i in range(len(self.actions_spaces)):
                if len(self.actions_spaces[i].sample()) == 3:
                    a.append([0, 0.1, 0.1])
                elif len(self.actions_spaces[
                             i].sample()) == 2:  # heat pump action is always considered, therefore if i have number of actions = 2 i have one storage and the heat pump
                    a.append([0, 0.1])
                else:
                    a.append([0.1])
        self.action_tracker.append(a)
        return np.array(a)


def auto_size(buildings):
    for building in buildings.values():

        # Autosize guarantees that the DHW device is large enough to always satisfy the maximum DHW demand
        if building.dhw_heating_device.nominal_power == 'autosize':

            # If the DHW device is a HeatPump
            if isinstance(building.dhw_heating_device, HeatPump):

                # We assume that the heat pump is always large enough to meet the highest heating or cooling demand of the building
                building.dhw_heating_device.nominal_power = np.array(
                    building.sim_results['dhw_demand'] / building.dhw_heating_device.cop_heating).max()

            # If the device is an electric heater
            elif isinstance(building.dhw_heating_device, ElectricHeater):
                building.dhw_heating_device.nominal_power = (
                            np.array(building.sim_results['dhw_demand']) / building.dhw_heating_device.efficiency).max()

        # Autosize guarantees that the cooling device device is large enough to always satisfy the maximum DHW demand
        if building.cooling_device.nominal_power == 'autosize':
            building.cooling_device.nominal_power = safety_factor *(np.array(building.sim_results['q_cooling'])).max()

        # Defining the capacity of the storage devices as a number of times the maximum demand
        building.dhw_storage.capacity = max(building.sim_results['dhw_demand']) * building.dhw_storage.capacity
        building.cooling_storage.capacity = max(
            building.sim_results['cooling_demand']) * building.cooling_storage.capacity

        # Done in order to avoid dividing by 0 if the capacity is 0
        if building.dhw_storage.capacity <= 0.00001:
            building.dhw_storage.capacity = 0.00001
        if building.cooling_storage.capacity <= 0.00001:
            building.cooling_storage.capacity = 0.00001

def building_loader(data_path, building_attributes, weather_file, solar_profile, el_data, building_ids,
                    buildings_states_actions, simulation_period=(0, 8759)):


    with open(building_attributes) as json_file:
        data = json.load(json_file)




    buildings, observation_spaces, action_spaces = {}, [], []
    s_low_central_agent, s_high_central_agent, appended_states = [], [], []
    a_low_central_agent, a_high_central_agent, appended_actions = [], [], []

    for uid, attributes in zip(data, data.values()):
        index = int(uid.split("_")[-1])  #Building_n  --> extract n and convert it into  an int  --> this allows to pick always the correct row in min and max dataframe, independently on the number of buildings considered
        if uid in building_ids:
            heat_pump = HeatPump(nominal_power=attributes['Heat_Pump']['nominal_power'],
                                 eta_tech=attributes['Heat_Pump']['technical_efficiency'],
                                 t_target_heating=attributes['Heat_Pump']['t_target_heating'],
                                 t_target_cooling=attributes['Heat_Pump']['t_target_cooling'])

            electric_heater = ElectricHeater(nominal_power=attributes['Electric_Water_Heater']['nominal_power'],
                                             efficiency=attributes['Electric_Water_Heater']['efficiency'])

            chilled_water_tank = EnergyStorage(capacity=attributes['Chilled_Water_Tank']['capacity'],
                                               loss_coeff=attributes['Chilled_Water_Tank']['loss_coefficient'])

            dhw_tank = EnergyStorage(capacity=attributes['DHW_Tank']['capacity'],
                                     loss_coeff=attributes['DHW_Tank']['loss_coefficient'])

            model_building = BuildingDynamics(n_features=attributes["Model_dynamics"]["n_features"],
                                              lookback=attributes["Model_dynamics"]["lookback"],
                                              n_hidden=attributes["Model_dynamics"]["n_hidden"],
                                              n_layers=attributes["Model_dynamics"]["n_layers"])

            building = Building(buildingId=uid, dhw_storage=dhw_tank, cooling_storage=chilled_water_tank,
                                dhw_heating_device=electric_heater, cooling_device=heat_pump, model_dynamics=model_building)

            lookback = attributes["Model_dynamics"]["lookback"]

            data_file = str(uid) + '.csv'
            simulation_data = data_path / data_file
            with open(simulation_data) as csv_file:
                data = pd.read_csv(csv_file)

            building.sim_results['cooling_demand'] = list(
                data['Cooling Load [kWh]'][simulation_period[0]:simulation_period[1] + 1])
            building.sim_results['dhw_demand'] = list(
                data['DHW Heating [kWh]'][simulation_period[0]:simulation_period[1] + 1])
            building.sim_results['non_shiftable_load'] = list(
                data['Equipment Electric Power [kWh]'][simulation_period[0]:simulation_period[1] + 1])
            building.sim_results['month'] = list(data['Month'][simulation_period[0]:simulation_period[1] + 1])
            building.sim_results['day'] = list(data['Day Type'][simulation_period[0]:simulation_period[1] + 1])
            building.sim_results['hour'] = list(data['Hour'][simulation_period[0]:simulation_period[1] + 1])
            building.sim_results['daylight_savings_status'] = list(
                data['Daylight Savings Status'][simulation_period[0]:simulation_period[1] + 1])
            building.sim_results['t_in'] = list(
                data['Indoor Temperature [C]'][simulation_period[0]:simulation_period[1] + 1])
            building.sim_results['avg_unmet_setpoint'] = list(
                data['Average Unmet Cooling Setpoint Difference [C]'][simulation_period[0]:simulation_period[1] + 1])
            building.sim_results['rh_in'] = list(
                data['Indoor Relative Humidity [%]'][simulation_period[0]:simulation_period[1] + 1])
            building.sim_results['occupancy'] = list(
                data['Occupancy'][simulation_period[0]:simulation_period[1] + 1])

            building.sim_results['q_cooling'] = list(
                data['Cooling Load [kWh]'][simulation_period[0]:simulation_period[1] + 1])
            building.sim_results['set_point'] = list(
                data['Set Point'][simulation_period[0]:simulation_period[1] + 1])

            building.sim_results['Time_high_price'] = list(
                data['time_high'][simulation_period[0]:simulation_period[1] + 1])
            building.sim_results['Time_low_price'] = list(
                data['time_low'][simulation_period[0]:simulation_period[1] + 1])

#=======================================================================================================#
            #Set building.lstm_results['T_lstm'] as a list of a generic parameter of data and then overwrite time step by time step

            building.lstm_results['T_lstm'] = list(
                data['Cooling Load [kWh]'][simulation_period[0]:simulation_period[1] + 1])
            building.sim_results['Q_action'] = np.zeros([simulation_period[1]+1-simulation_period[0],1]).tolist()

            with open(weather_file) as csv_file:
                weather_data = pd.read_csv(csv_file)

            building.sim_results['t_out'] = list(
                 weather_data['Outdoor Drybulb Temperature [C]'][simulation_period[0]:simulation_period[1] + 1])
            building.sim_results['rh_out'] = list(
                weather_data['Outdoor Relative Humidity [%]'][simulation_period[0]:simulation_period[1] + 1])
            building.sim_results['diffuse_solar_rad'] = list(
                weather_data['Diffuse Solar Radiation [W/m2]'][simulation_period[0]:simulation_period[1] + 1])
            building.sim_results['direct_solar_rad'] = list(
                weather_data['Direct Solar Radiation [W/m2]'][simulation_period[0]:simulation_period[1] + 1])

            # Reading weather forecasts
            building.sim_results['t_out_pred_6h'] = list(weather_data['6h Prediction Outdoor Drybulb Temperature [C]'][
                                                         simulation_period[0]:simulation_period[1] + 1])
            building.sim_results['t_out_pred_12h'] = list(
                weather_data['12h Prediction Outdoor Drybulb Temperature [C]'][
                simulation_period[0]:simulation_period[1] + 1])
            building.sim_results['t_out_pred_24h'] = list(
                weather_data['24h Prediction Outdoor Drybulb Temperature [C]'][
                simulation_period[0]:simulation_period[1] + 1])

            building.sim_results['rh_out_pred_6h'] = list(weather_data['6h Prediction Outdoor Relative Humidity [%]'][
                                                          simulation_period[0]:simulation_period[1] + 1])
            building.sim_results['rh_out_pred_12h'] = list(weather_data['12h Prediction Outdoor Relative Humidity [%]'][
                                                           simulation_period[0]:simulation_period[1] + 1])
            building.sim_results['rh_out_pred_24h'] = list(weather_data['24h Prediction Outdoor Relative Humidity [%]'][
                                                           simulation_period[0]:simulation_period[1] + 1])

            building.sim_results['diffuse_solar_rad_pred_6h'] = list(
                weather_data['6h Prediction Diffuse Solar Radiation [W/m2]'][
                simulation_period[0]:simulation_period[1] + 1])
            building.sim_results['diffuse_solar_rad_pred_12h'] = list(
                weather_data['12h Prediction Diffuse Solar Radiation [W/m2]'][
                simulation_period[0]:simulation_period[1] + 1])
            building.sim_results['diffuse_solar_rad_pred_24h'] = list(
                weather_data['24h Prediction Diffuse Solar Radiation [W/m2]'][
                simulation_period[0]:simulation_period[1] + 1])

            building.sim_results['direct_solar_rad_pred_6h'] = list(
                weather_data['6h Prediction Direct Solar Radiation [W/m2]'][
                simulation_period[0]:simulation_period[1] + 1])
            building.sim_results['direct_solar_rad_pred_12h'] = list(
                weather_data['12h Prediction Direct Solar Radiation [W/m2]'][
                simulation_period[0]:simulation_period[1] + 1])
            building.sim_results['direct_solar_rad_pred_24h'] = list(
                weather_data['24h Prediction Direct Solar Radiation [W/m2]'][
                simulation_period[0]:simulation_period[1] + 1])


            # ===================================================================================================
            # Extract lagged variables in order to feed the LSTM models
            building.sim_results_lagged['occupancy'] = list(
                data['Occupancy'][simulation_period[0] - lookback + 1: simulation_period[1] + 1])
            building.sim_results_lagged['q_cooling'] = list(
                data['Cooling Load [kWh]'][simulation_period[0] - lookback + 1: simulation_period[1] + 1])
            building.sim_results_lagged['month'] = list(
                data['Month'][simulation_period[0] - lookback + 1:simulation_period[1] + 1])
            building.sim_results_lagged['day'] = list(
                data['Day Type'][simulation_period[0] - lookback + 1:simulation_period[1] + 1])
            building.sim_results_lagged['hour'] = list(
                data['Hour'][simulation_period[0] - lookback + 1:simulation_period[1] + 1])
            building.sim_results_lagged['t_in'] = list(
                data['Indoor Temperature [C]'][
                simulation_period[0] - lookback:simulation_period[1]])  # i can control T at timestep t+1
            building.sim_results_lagged['direct_solar_rad'] = list(
                weather_data['Direct Solar Radiation [W/m2]'][simulation_period[0]-lookback+1:simulation_period[1] +1])
            building.sim_results_lagged['t_out'] = list(
                weather_data['Outdoor Drybulb Temperature [C]'][simulation_period[0]-lookback+1:simulation_period[1] +1])

            # Reading the building attributes
            building.building_type = attributes['Building_Type']
            building.climate_zone = attributes['Climate_Zone']
            building.solar_power_capacity = attributes['Solar_Power_Installed(kW)']

            with open(solar_profile) as csv_file:
                solar_data = pd.read_csv(csv_file)

            building.sim_results['solar_gen'] = list(
                attributes['Solar_Power_Installed(kW)'] * solar_data['Hourly Data: AC inverter power (W)'][simulation_period[0]:simulation_period[1] + 1] / 1000)

            #Load electricity price
            with open(el_data) as csv_file:
                electricity_price = pd.read_csv(csv_file)

            building.sim_results['el_price'] = list(
                electricity_price['Price'][simulation_period[0]:simulation_period[1] + 1])
            building.sim_results['el_price_pred_1h'] = list(np.roll(
                electricity_price['Price'][simulation_period[0]:simulation_period[1] + 1], -1))

            building.sim_results['el_price_pred_2h'] = list(np.roll(
                electricity_price['Price'][simulation_period[0]:simulation_period[1] + 1], -2))

            building.sim_results['el_price_pred_3h'] = list(np.roll(
                electricity_price['Price'][simulation_period[0]:simulation_period[1] + 1], -3))

            building.sim_results['cop_T'] = []
            for i in range(simulation_period[1]+1-simulation_period[0]):
                cop_t = cop_T(building.sim_results['t_out'][i], heat_pump.nominal_COP)
                building.sim_results['cop_T'].append(cop_t)

            # Finding the max and min possible values of all the states, which can then be used by the RL agent to scale the states and train any function approximators more effectively
            s_low, s_high = [], []
            action_dependent_states = ['cooling_storage_soc','dhw_storage_soc','cop','district_power','T_lstm','deltaT','q_cooling']
            for state_name, value in zip(buildings_states_actions[uid]['states'],
                                         buildings_states_actions[uid]['states'].values()):

                if value == True:
                    if state_name not in action_dependent_states:
                        s_low.append(min(building.sim_results[state_name]))
                        s_high.append(max(building.sim_results[state_name]))

                        # Create boundaries of the observation space of a centralized agent (if a central agent is being used instead of decentralized ones). We include all the weather variables used as states, and use the list appended_states to make sure we don't include any repeated states (i.e. weather variables measured by different buildings)
                        if state_name in ['t_in', 'avg_unmet_setpoint', 'rh_in', 'non_shiftable_load', 'solar_gen']:
                            s_low_central_agent.append(min(building.sim_results[state_name]))
                            s_high_central_agent.append(max(building.sim_results[state_name]))

                        elif state_name not in appended_states:
                            s_low_central_agent.append(min(building.sim_results[state_name]))
                            s_high_central_agent.append(max(building.sim_results[state_name]))
                            appended_states.append(state_name)

                    elif state_name == 'cop':
                        s_low.append(0.0)
                        s_high.append(max(building.sim_results['cop_T']))
                        s_low_central_agent.append(0.0)
                        s_high_central_agent.append(max(building.sim_results['cop_T']))

                    elif state_name == 'district_power':
                        s_low.append(0.0)
                        s_high.append(300) #maximum district consumption
                        s_low_central_agent.append(0.0)
                        s_high_central_agent.append(300)

                    elif state_name == 'T_lstm':
                        minT = 20 #min_value.iloc[index-1,-1]
                        maxT = 32 #max_value.iloc[index-1,-1]
                        s_low.append(minT)
                        s_high.append(maxT)
                        s_low_central_agent.append(minT)
                        s_high_central_agent.append(maxT)

                    elif state_name == 'deltaT':
                        s_low.append(0)
                        s_high.append(3)
                        s_low_central_agent.append(0)
                        s_high_central_agent.append(3)

                    elif state_name == 'q_cooling':
                        s_low.append(0)
                        s_high.append(safety_factor*(np.array(building.sim_results['q_cooling'])).max())
                        s_low_central_agent.append(0)
                        s_high_central_agent.append(safety_factor*(np.array(building.sim_results['q_cooling'])).max())

                    else:
                        s_low.append(0.0)
                        s_high.append(1.0)
                        s_low_central_agent.append(0.0)
                        s_high_central_agent.append(1.0)

            '''The energy storage (tank) capacity indicates how many times bigger the tank is compared to the maximum hourly energy demand of the building (cooling or DHW respectively), which sets a lower bound for the action of 1/tank_capacity, as the energy storage device can't provide the building with more energy than it will ever need for a given hour. The heat pump is sized using approximately the maximum hourly energy demand of the building (after accounting for the COP, see function autosize). Therefore, we make the fair assumption that the action also has an upper bound equal to 1/tank_capacity. This boundaries should speed up the learning process of the agents and make them more stable rather than if we just set them to -1 and 1. I.e. if Chilled_Water_Tank.Capacity is 3 (3 times the max. hourly demand of the building in the entire year), its actions will be bounded between -1/3 and 1/3'''
            a_low, a_high = [], []

            for action_name, value in zip(buildings_states_actions[uid]['actions'],
                                          buildings_states_actions[uid]['actions'].values()):
                if value == True:

                    if action_name == 'heat_pump_to_buildng':
                            a_low.append(0)
                            a_high.append(1)
                            a_low_central_agent.append(0)
                            a_high_central_agent.append(1)


                    elif action_name == 'cooling_storage':

                        # Avoid division by 0
                        if attributes['Chilled_Water_Tank']['capacity'] > 0.000000001:
                            a_low.append(max(-1.0 / attributes['Chilled_Water_Tank']['capacity'], -1.0))
                            a_high.append(min(1.0 / attributes['Chilled_Water_Tank']['capacity'], 1.0))
                            a_low_central_agent.append(max(-1.0 / attributes['Chilled_Water_Tank']['capacity'], -1.0))
                            a_high_central_agent.append(min(1.0 / attributes['Chilled_Water_Tank']['capacity'], 1.0))
                        else:
                            a_low.append(-1.0)
                            a_high.append(1.0)
                            a_low_central_agent.append(-1.0)
                            a_high_central_agent.append(1.0)
                    else:
                        if attributes['DHW_Tank']['capacity'] > 0.000000001:
                            a_low.append(max(-1.0 / attributes['DHW_Tank']['capacity'], -1.0))
                            a_high.append(min(1.0 / attributes['DHW_Tank']['capacity'], 1.0))
                            a_low_central_agent.append(max(-1.0 / attributes['DHW_Tank']['capacity'], -1.0))
                            a_high_central_agent.append(min(1.0 / attributes['DHW_Tank']['capacity'], 1.0))
                        else:
                            a_low.append(-1.0)
                            a_high.append(1.0)
                            a_low_central_agent.append(-1.0)
                            a_high_central_agent.append(1.0)

            building.set_state_space(np.array(s_high), np.array(s_low))
            building.set_action_space(np.array(a_high), np.array(a_low))

            observation_spaces.append(building.observation_space)
            action_spaces.append(building.action_space)

            buildings[uid] = building

    observation_space_central_agent = spaces.Box(low=np.array(s_low_central_agent), high=np.array(s_high_central_agent),
                                                 dtype=np.float32)

    action_space_central_agent = spaces.Box(low=np.array(a_low_central_agent), high=np.array(a_high_central_agent),
                                            dtype=np.float32)

    for building in buildings.values():

        # If the DHW device is a HeatPump
        if isinstance(building.dhw_heating_device, HeatPump):
            # Calculating COPs of the heat pumps for every hour
            building.dhw_heating_device.cop_heating = building.dhw_heating_device.eta_tech * (
                        building.dhw_heating_device.t_target_heating + 273.15) / (
                                                                  building.dhw_heating_device.t_target_heating -
                                                                  weather_data['Outdoor Drybulb Temperature [C]'])
            building.dhw_heating_device.cop_heating[building.dhw_heating_device.cop_heating < 0] = 20.0
            building.dhw_heating_device.cop_heating[building.dhw_heating_device.cop_heating > 20] = 20.0
            building.dhw_heating_device.cop_heating = building.dhw_heating_device.cop_heating.to_numpy()

    auto_size(buildings)

    return buildings, observation_spaces, action_spaces, observation_space_central_agent, action_space_central_agent

class CityLearn_3dem(gym.Env):
    def __init__(self, data_path, building_attributes, weather_file, solar_profile, el_data, building_ids,
                 buildings_states_actions=None, building_dynamics_state=None, simulation_period=(0, 8759),
                 cost_function=['ramping', '1-load_factor', 'average_daily_peak', 'peak_demand',
                                'net_electricity_consumption'], central_agent=True, verbose=0):
        with open(buildings_states_actions) as json_file:
            self.buildings_states_actions = json.load(json_file)
        with open(building_dynamics_state) as json_file:
            self.building_dynamics_state = json.load(json_file)
        hidden = []
        self.hidden_list = hidden
        self.buildings_states_actions_filename = buildings_states_actions
        self.building_dynamics_state_filename = building_dynamics_state
        self.building_attributes = building_attributes
        self.solar_profile = solar_profile
        self.el_data = el_data
        self.building_ids = building_ids
        self.cost_function = cost_function
        self.cost_rbc = None
        self.data_path = data_path
        self.weather_file = weather_file
        self.central_agent = central_agent
        self.loss = []
        self.verbose = verbose
        self.simulation_period = simulation_period

        self.buildings, self.observation_spaces, self.action_spaces, self.observation_space, self.action_space = building_loader(
            data_path, building_attributes, weather_file, solar_profile, el_data, building_ids,
            self.buildings_states_actions, simulation_period=self.simulation_period)
        self.uid = None
        self.n_buildings = len([i for i in self.buildings])
        self.reset()
        for uid, building in self.buildings.items():
            models = building.model_dynamics
            models.load_state_dict(torch.load('Building_models/'+ uid + str('.pth')))
            hi = models.init_hidden(1)
            hidden.append(hi)
    def get_state_action_spaces(self):
        return self.observation_spaces, self.action_spaces

    def next_hour(self):
        self.time_step = next(self.hour)
        for building in self.buildings.values():
            building.time_step = self.time_step

    def get_building_information(self):

        np.seterr(divide='ignore', invalid='ignore')
        # Annual DHW demand, Annual Cooling Demand, Annual Electricity Demand
        building_info = {}
        for uid, building in self.buildings.items():
            building_info[uid] = {}
            building_info[uid]['building_type'] = building.building_type
            building_info[uid]['climate_zone'] = building.climate_zone
            building_info[uid]['solar_power_capacity (kW)'] = round(building.solar_power_capacity, 3)
            building_info[uid]['Annual_DHW_demand (kWh)'] = round(sum(building.sim_results['dhw_demand']), 3)
            building_info[uid]['Annual_cooling_demand (kWh)'] = round(sum(building.sim_results['cooling_demand']), 3)
            building_info[uid]['Annual_nonshiftable_electrical_demand (kWh)'] = round(
                sum(building.sim_results['non_shiftable_load']), 3)

            building_info[uid]['Correlations_DHW'] = {}
            building_info[uid]['Correlations_cooling_demand'] = {}
            building_info[uid]['Correlations_non_shiftable_load'] = {}

            for uid_corr, building_corr in self.buildings.items():
                if uid_corr != uid:
                    building_info[uid]['Correlations_DHW'][uid_corr] = round((np.corrcoef(
                        np.array(building.sim_results['dhw_demand']),
                        np.array(building_corr.sim_results['dhw_demand'])))[0][1], 3)
                    building_info[uid]['Correlations_cooling_demand'][uid_corr] = round((np.corrcoef(
                        np.array(building.sim_results['cooling_demand']),
                        np.array(building_corr.sim_results['cooling_demand'])))[0][1], 3)
                    building_info[uid]['Correlations_non_shiftable_load'][uid_corr] = round((np.corrcoef(
                        np.array(building.sim_results['non_shiftable_load']),
                        np.array(building_corr.sim_results['non_shiftable_load'])))[0][1], 3)

        return building_info

    def step(self, actions):  

        district_consumption = []
        electric_demand = 0
        elec_consumption_dhw_storage = 0
        elec_consumption_cooling_storage = 0
        elec_consumption_dhw_total = 0
        elec_consumption_cooling_total = 0
        elec_consumption_appliances = 0
        elec_generation = 0
        temperature = []
        Q_cool = []

        if self.central_agent:
            # If the agent is centralized, all the actions for all the buildings are provided as an ordered list of numbers. The order corresponds to the order of the buildings as they appear on the file building_attributes.json, and only considering the buildings selected for the simulation by the user (building_ids).
            count = 0
            for uid, building in self.buildings.items():
                #load dynamic models
                models = building.model_dynamics
                models.load_state_dict(torch.load('Building_models/' + uid + str('.pth')))
                lookback = building.model_dynamics.lookback
                n_features = building.model_dynamics.n_features

                nn_dynamics = pd.DataFrame(np.zeros((lookback, n_features)))
                index = int(uid.split("_")[-1]) #buildings should be runned in ascending order
                building_electric_demand = 0

                if self.buildings_states_actions[uid]['actions']['heat_pump_to_buildng']:
                    #setback when non-occupied
                    if  round(building.sim_results['set_point'][self.time_step]) == 30:
                        actions[0]=0
                        building.sim_results['cooling_demand'][self.time_step]= actions[0]

                    else:
                        building.sim_results['cooling_demand'][self.time_step] = min(actions[0] * building.cooling_device.nominal_power,
                          cop_curve_cooling(actions[0] * building.cooling_device.nominal_power,
                                            building.sim_results['t_out'][self.time_step],
                                            building.cooling_device.nominal_power, building.cooling_device.nominal_COP)[1])

                    actions = actions[1:]

                    #-----------------------nn_dynamics starting point
                    building.sim_results['Q_action'][self.time_step] = building.sim_results['cooling_demand'][self.time_step]
                    T_in = building.sim_results_lagged["t_in"][self.time_step:self.time_step + lookback]
                    Q_cooling = building.sim_results_lagged["q_cooling"][self.time_step:self.time_step + lookback]
                    if self.time_step < lookback:

                        Q_input = np.concatenate(
                            [Q_cooling[0:lookback-1 - self.time_step], building.sim_results['Q_action'][0:self.time_step + 1]])
                        if self.time_step == 0:
                            T_lag = T_in
                        else:
                            T_lag = np.concatenate([T_in[0: lookback - self.time_step],
                                                    building.lstm_results['T_lstm'][0:self.time_step]])
                    else:
                        Q_input = building.sim_results['Q_action'][(self.time_step - lookback+1):self.time_step+1]
                        T_lag = np.array(building.lstm_results['T_lstm'][self.time_step - lookback:self.time_step])


                    column = ['Direct_Solar_Rad', 'T_ext', 'Occupants', 'Q_cooling', 'sinhour', 'coshour', 'sinday',
                              'cosday', 'sinmonth', 'cosmonth', 'T_int']
                    nn_dynamics.columns = column
                    timestamp_modifier = ['hour','day','month']
                    artificial_data = ['t_in','q_cooling']
                    modified_state = timestamp_modifier + artificial_data
                    dynamics_to_data = {'occupancy': 'Occupants',  'direct_solar_rad': 'Direct_Solar_Rad',
                                        't_out': 'T_ext'}
                    for dynamic_state_name, value in zip(self.building_dynamics_state[uid],self.building_dynamics_state[uid].values()):
                        if value == True:
                            if dynamic_state_name not in modified_state:
                                nn_dynamics[dynamics_to_data[dynamic_state_name]] = building.sim_results_lagged[dynamic_state_name][
                                                                  self.time_step:self.time_step + lookback]
                            elif dynamic_state_name in timestamp_modifier:
                                nn_dynamics['sinhour'] = np.sin(2 * np.pi * np.array(building.sim_results_lagged["hour"][self.time_step:self.time_step + lookback]) / 24)
                                nn_dynamics['coshour'] = np.cos(2 * np.pi * np.array(building.sim_results_lagged["hour"][self.time_step:self.time_step + lookback]) / 24)
                                nn_dynamics['sinday'] = np.sin(2 * np.pi * np.array(building.sim_results_lagged["day"][self.time_step:self.time_step + lookback]) / 7)
                                nn_dynamics['cosday'] = np.cos(2 * np.pi * np.array(building.sim_results_lagged["day"][self.time_step:self.time_step + lookback]) / 7)
                                nn_dynamics['sinmonth'] = np.sin(2 * np.pi * np.array(building.sim_results_lagged["month"][self.time_step:self.time_step + lookback]) / 12)
                                nn_dynamics['cosmonth'] = np.cos(2 * np.pi * np.array(building.sim_results_lagged["month"][self.time_step:self.time_step + lookback]) / 12)
                            elif dynamic_state_name in artificial_data:
                                nn_dynamics['Q_cooling'] = Q_input
                                nn_dynamics['T_int'] = T_lag
                    nn_dynamics = np.array(nn_dynamics)
                    nn_scaled = np.zeros((len(nn_dynamics), len(nn_dynamics[:][0])))

                    h = self.hidden_list[index-1]

                    # using the two csv normalize the input of the nn (only works with consecutive building indexes)
                    max_norm = np.array(max_value.iloc[index - 1, :])
                    min_norm = np.array(min_value.iloc[index - 1, :])
                    for z in range(0, len(nn_dynamics)):
                        num = np.subtract(nn_dynamics[z, :], min_norm)
                        denom = np.subtract(max_norm, min_norm)
                        nn_scaled[z, :] = np.divide(num, denom)
                    inputs = torch.tensor(nn_scaled, dtype=torch.float32)
                    inputs = inputs[np.newaxis, :, :]
                    h = tuple([each.data for each in h])

                    test_output, h = models(inputs.float(),h)

                    T_in_lstm = test_output * (max_norm[-1] - min_norm[-1]) + min_norm[-1]

                    building.lstm_results['T_lstm'][self.time_step] = T_in_lstm.item()

                    self.hidden_list[index-1] = h
                    count = count + 1

                # -----------------------nn_dynamics ending point
                # -----------------------focus on storage actions
                if self.buildings_states_actions[uid]['actions']['cooling_storage']:
                    # Cooling
                    _electric_demand_cooling = building.set_storage_cooling(actions[0])
                    self.cooling_stor_actions = actions[0]
                    actions = actions[1:]

                    elec_consumption_cooling_storage += building._electric_consumption_cooling_storage
                else:
                    _electric_demand_cooling = 0

                if self.buildings_states_actions[uid]['actions']['dhw_storage']:
                    # DHW
                    _electric_demand_dhw = building.set_storage_heating(actions[0])
                    self.dhw_stor_actions = actions[0]
                    actions = actions[1:]
                    elec_consumption_dhw_storage += building._electric_consumption_dhw_storage
                else:
                    _electric_demand_dhw = 0

                # Total heating and cooling electrical loads
                elec_consumption_cooling_total += _electric_demand_cooling
                elec_consumption_dhw_total += _electric_demand_dhw

                # Electrical appliances
                _non_shiftable_load = building.get_non_shiftable_load()
                elec_consumption_appliances += _non_shiftable_load

                # Solar generation
                _solar_generation = building.get_solar_power()
                elec_generation += _solar_generation

                # Adding loads from appliances and subtracting solar generation to the net electrical load of each building
                building_electric_demand += _electric_demand_cooling + _electric_demand_dhw + _non_shiftable_load - _solar_generation

                # Electricity consumed by buildings
                district_consumption.append(-building_electric_demand)
                # Total electricity consumption
                electric_demand += building_electric_demand
                temperature.append(building.lstm_results['T_lstm'][self.time_step])
                Q_cool.append(building.sim_results['cooling_demand'][self.time_step])


            assert len(actions) == 0, 'Some of the actions provided were not used'

        else: #DECENTRALIZED CONFIGURATION

            assert len(
                actions) == self.n_buildings, "The length of the list of actions should match the length of the list of buildings."
            count = 0
            for a, (uid, building) in zip(actions, self.buildings.items()):
                models = building.model_dynamics
                models.load_state_dict(torch.load('Building_models/'+ uid + str('.pth')))
                index = int(uid.split("_")[-1])
                lookback = building.model_dynamics.lookback
                n_features = building.model_dynamics.n_features
                nn_dynamics = pd.DataFrame(np.zeros((lookback, n_features)))
                assert sum(self.buildings_states_actions[uid]['actions'].values()) == len(
                    a), "The number of input actions for building " + str(
                    uid) + " must match the number of actions defined in the list of building attributes."

                building_electric_demand = 0

                if self.buildings_states_actions[uid]['actions']['heat_pump_to_buildng']:
                    Q_cooling = building.sim_results_lagged["q_cooling"][self.time_step:self.time_step + lookback]
                    T_in = building.sim_results_lagged["t_in"][self.time_step:self.time_step + lookback]
                    building.sim_results['Q_action'][self.time_step] = building.sim_results['cooling_demand'][
                        self.time_step]

                    #-----------------------nn_dynamics starting point
                    if self.time_step < lookback:

                        Q_input = np.concatenate(
                            [Q_cooling[0:lookback-1 - self.time_step], building.sim_results['Q_action'][0:self.time_step + 1]])
                        if self.time_step == 0:
                            T_lag = T_in
                        else:
                            T_lag = np.concatenate([T_in[0: lookback - self.time_step ],
                                                    building.lstm_results['T_lstm'][0:self.time_step ]])
                    else:
                        Q_input = building.sim_results['Q_action'][(self.time_step - lookback+1):self.time_step+1]
                        T_lag = np.array(building.lstm_results['T_lstm'][self.time_step - lookback :self.time_step])

                    column = ['Direct_Solar_Rad', 'T_ext', 'Occupants', 'Q_cooling', 'sinhour', 'coshour', 'sinday',
                              'cosday', 'sinmonth', 'cosmonth', 'T_int']
                    nn_dynamics.columns = column
                    timestamp_modifier = ['hour', 'day', 'month']
                    artificial_data = ['t_in', 'Q_cooling']
                    modified_state = timestamp_modifier + artificial_data
                    dynamics_to_data = {'occupancy': 'Occupants', 'direct_solar_rad': 'Direct_Solar_Rad', 'q_cooling':'Q_cooling',
                                        't_out': 'T_ext'}
                    for dynamic_state_name, value in zip(self.building_dynamics_state[uid],
                                                         self.building_dynamics_state[uid].values()):
                        if value == True:
                            if dynamic_state_name not in modified_state:
                                nn_dynamics[dynamics_to_data[dynamic_state_name]] = building.sim_results_lagged[dynamic_state_name][
                                                                  self.time_step:self.time_step + lookback]
                            elif dynamic_state_name in timestamp_modifier:
                                nn_dynamics['sinhour'] = np.sin(2 * np.pi * np.array(
                                    building.sim_results_lagged["hour"][self.time_step:self.time_step + lookback]) / 24)
                                nn_dynamics['coshour'] = np.cos(2 * np.pi * np.array(
                                    building.sim_results_lagged["hour"][self.time_step:self.time_step + lookback]) / 24)
                                nn_dynamics['sinday'] = np.sin(2 * np.pi * np.array(
                                    building.sim_results_lagged["day"][self.time_step:self.time_step + lookback]) / 7)
                                nn_dynamics['cosday'] = np.cos(2 * np.pi * np.array(
                                    building.sim_results_lagged["day"][self.time_step:self.time_step + lookback]) / 7)
                                nn_dynamics['sinmonth'] = np.sin(2 * np.pi * np.array(building.sim_results_lagged["month"][
                                                                             self.time_step:self.time_step + lookback]) / 12)
                                nn_dynamics['cosmonth'] = np.cos(2 * np.pi * np.array(building.sim_results_lagged["month"][
                                                                             self.time_step:self.time_step + lookback]) / 12)
                            elif dynamic_state_name in artificial_data:
                                nn_dynamics['Q_cooling'] = Q_input
                                nn_dynamics['T_int'] = T_lag
                    nn_dynamics = np.array(nn_dynamics)
                    nn_scaled = np.zeros((len(nn_dynamics), len(nn_dynamics[:][0])))

                    h = self.hidden_list[index-1]
                    # using the two csv normalize the input of the nn (only works with consecutive building indexes)
                    max_norm = np.array(max_value.iloc[index - 1, :])
                    min_norm = np.array(min_value.iloc[index - 1, :])
                    for z in range(0, len(nn_dynamics)):
                        num = np.subtract(nn_dynamics[z, :], min_norm)
                        denom = np.subtract(max_norm, min_norm)
                        nn_scaled[z, :] = np.divide(num, denom)
                    inputs = torch.tensor(nn_scaled, dtype=torch.float32)
                    inputs = inputs[np.newaxis, :, :]
                    h = tuple([each.data for each in h])
                    test_output, h = models(inputs.float(),h)

                    T_in_lstm = test_output * (max_norm[-1] - min_norm[-1]) + min_norm[-1]
                    building.lstm_results['T_lstm'][self.time_step] = T_in_lstm.item()
                    self.hidden_list[index-1] = h
                    count = count + 1
                    # -----------------------nn_dynamics ending point
                    # -----------------------focus on storage actions
                    if self.buildings_states_actions[uid]['actions']['cooling_storage']:
                        # Cooling
                        _electric_demand_cooling = building.set_storage_cooling(a[1])
                        elec_consumption_cooling_storage += building._electric_consumption_cooling_storage

                        if self.buildings_states_actions[uid]['actions']['dhw_storage']:
                            # DHW
                            _electric_demand_dhw = building.set_storage_heating(a[2])
                            elec_consumption_dhw_storage += building._electric_consumption_dhw_storage

                        else:

                            _electric_demand_dhw = 0

                    else:
                        _electric_demand_cooling = 0
                        # check if there is a DHW storage
                        if self.buildings_states_actions[uid]['actions']['dhw_storage']:
                            # DHW
                            _electric_demand_dhw = building.set_storage_heating(a[1])
                            elec_consumption_dhw_storage += building._electric_consumption_dhw_storage

                        else:

                            _electric_demand_dhw = 0
                else:
                    if self.buildings_states_actions[uid]['actions']['cooling_storage']:
                        # Cooling
                        _electric_demand_cooling = building.set_storage_cooling(a[0])
                        elec_consumption_cooling_storage += building._electric_consumption_cooling_storage

                        if self.buildings_states_actions[uid]['actions']['dhw_storage']:
                            # DHW
                            _electric_demand_dhw = building.set_storage_heating(a[1])
                            elec_consumption_dhw_storage += building._electric_consumption_dhw_storage

                        else:

                            _electric_demand_dhw = 0

                    else:
                        _electric_demand_cooling = 0
                        _electric_demand_dhw = building.set_storage_heating(a[0])
                        elec_consumption_dhw_storage += building._electric_consumption_dhw_storage
                # Total heating and cooling electrical loads
                elec_consumption_cooling_total += _electric_demand_cooling
                elec_consumption_dhw_total += _electric_demand_dhw

                # Electrical appliances
                _non_shiftable_load = building.get_non_shiftable_load()
                elec_consumption_appliances += _non_shiftable_load

                # Solar generation
                _solar_generation = building.get_solar_power()
                elec_generation += _solar_generation

                # Adding loads from appliances and subtracting solar generation to the net electrical load of each building
                building_electric_demand += _electric_demand_cooling + _electric_demand_dhw + _non_shiftable_load - _solar_generation


                district_consumption.append(-building_electric_demand)
                # Total electricity consumption
                electric_demand += building_electric_demand

        self.next_hour()

        if self.central_agent:
            s, s_appended = [], []
            for uid, building in self.buildings.items():
                # If the agent is centralized, we append the states avoiding repetition. I.e. if multiple buildings share the outdoor temperature as a state, we only append it once to the states of the central agent. The variable s_appended is used for this purpose.
                for state_name, value in self.buildings_states_actions[uid]['states'].items():
                    if value == True:
                        if state_name not in s_appended:
                            if state_name in ['t_in', 'avg_unmet_setpoint', 'rh_in', 'non_shiftable_load', 'solar_gen']:
                                s.append(building.sim_results[state_name][self.time_step])
                            # -----------------------action dependent states are assigned
                            elif state_name != 'cooling_storage_soc' and state_name != 'dhw_storage_soc' and state_name != 'cop' and state_name != 'district_power' and state_name != 'T_lstm'  and state_name != 'deltaT' and state_name!= 'q_cooling':
                                s.append(building.sim_results[state_name][self.time_step])
                                s_appended.append(state_name)
                            elif state_name == 'cooling_storage_soc':
                                s.append(building.cooling_storage._soc / building.cooling_storage.capacity)
                            elif state_name == 'dhw_storage_soc':
                                s.append(building.dhw_storage._soc / building.dhw_storage.capacity)
                            elif state_name == 'cop':
                                s.append(building.cooling_device._cop_cooling)
                            elif state_name == 'district_power':
                                s.append(electric_demand)
                            elif state_name == 'T_lstm':
                                s.append(building.lstm_results['T_lstm'][self.time_step-1])
                            elif state_name == 'deltaT':
                                if round(building.sim_results['set_point'][self.time_step-1])==26:
                                    Tdiff = abs(building.sim_results['set_point'][self.time_step-1]-building.lstm_results['T_lstm'][self.time_step-1])
                                else:
                                    Tdiff=0
                                s.append(Tdiff)
                            elif state_name == 'q_cooling':
                                s.append(building.sim_results['cooling_demand'][self.time_step-1])
            self.state = np.array(s)
            rewards = reward_function_sa(district_consumption) #temperature can be added as well as cooling power to tune the reward function
            self.cumulated_reward_episode += rewards

        else:
            # If the controllers are decentralized, we append all the states to each associated agent's list of states.
            self.state = []
            for a, (uid, building) in zip(actions, self.buildings.items()):
                s = []
                for state_name, value in self.buildings_states_actions[uid]['states'].items():
                    if value == True:
                        if state_name != 'cooling_storage_soc' and state_name != 'dhw_storage_soc' and state_name != 'cop' and state_name != 'district_power' and state_name != 'T_lstm' and state_name != 'deltaT'and state_name != 'q_cooling':
                            s.append(building.sim_results[state_name][self.time_step])
                        #-----------------------action dependent states are assigned
                        elif state_name == 'cooling_storage_soc':
                            s.append(building.cooling_storage._soc / building.cooling_storage.capacity)
                        elif state_name == 'dhw_storage_soc':
                            s.append(building.dhw_storage._soc / building.dhw_storage.capacity)
                        elif state_name == 'cop':
                            s.append(building.cooling_device._cop_cooling)
                        elif state_name == 'district_power':
                            s.append(electric_demand)
                        elif state_name == 'T_lstm':
                            s.append(building.lstm_results['T_lstm'][self.time_step - 1])
                        elif state_name == 'deltaT':
                            if round(building.sim_results['set_point'][self.time_step - 1]) == 26:
                                Tdiff = abs(building.sim_results['set_point'][self.time_step - 1] -
                                            building.lstm_results['T_lstm'][self.time_step - 1])
                            else:
                                Tdiff = 0
                            s.append(Tdiff)
                        elif state_name == 'q_cooling':
                            s.append(building.sim_results['cooling_demand'][self.time_step - 1])
                self.state.append(np.array(s))
            self.state = np.array(self.state)
            rewards = reward_function_ma(district_consumption)
            self.cumulated_reward_episode += sum(rewards)
        # Control variables which are used to display the results and the behavior of the buildings at the district level.
        self.net_electric_consumption.append(electric_demand)
        self.electric_consumption_dhw_storage.append(elec_consumption_dhw_storage)
        self.electric_consumption_cooling_storage.append(elec_consumption_cooling_storage)
        self.electric_consumption_dhw.append(elec_consumption_dhw_total)
        self.electric_consumption_cooling.append(elec_consumption_cooling_total)
        self.electric_consumption_appliances.append(elec_consumption_appliances)
        self.electric_generation.append(elec_generation)
        self.net_electric_consumption_no_storage.append(
            electric_demand - elec_consumption_cooling_storage - elec_consumption_dhw_storage)
        self.net_electric_consumption_no_pv_no_storage.append(
            electric_demand + elec_generation - elec_consumption_cooling_storage - elec_consumption_dhw_storage)
        self.district_power.append(district_consumption)
        terminal = self._terminal()
        return (self._get_ob(), rewards, terminal, {})

    def reset_baseline_cost(self):
        self.cost_rbc = None

    def reset(self):
        # Initialization of variables
        self.hour = iter(np.array(range(0, self.simulation_period[1] + 1 - self.simulation_period[0])))
        self.next_hour()
        self.net_electric_consumption = []
        self.net_electric_consumption_no_storage = []
        self.net_electric_consumption_no_pv_no_storage = []
        self.electric_consumption_dhw_storage = []
        self.electric_consumption_cooling_storage = []
        self.electric_consumption_dhw = []
        self.electric_consumption_cooling = []
        self.electric_consumption_appliances = []
        self.electric_generation = []
        self.district_power = []
        self.cumulated_reward_episode = 0
        action_dependent_state = ['cooling_storage_soc', 'dhw_storage_soc', 'cop', 'district_power', 'T_lstm', 'deltaT', 'q_cooling']
        if self.central_agent:
            s, s_appended = [], []
            for uid, building in self.buildings.items():
                for state_name, value in self.buildings_states_actions[uid]['states'].items():
                    if state_name not in s_appended:
                        if value == True:
                            if state_name in ['t_in', 'avg_unmet_setpoint', 'rh_in', 'non_shiftable_load', 'solar_gen']:
                                s.append(building.sim_results[state_name][self.time_step])
                            elif state_name not in action_dependent_state:
                                s.append(building.sim_results[state_name][self.time_step])
                                s_appended.append(state_name)
                            elif state_name in action_dependent_state:
                                s.append(0.0)
                building.reset()
            self.state = np.array(s)
        else:
            self.state = []
            for uid, building in self.buildings.items():
                s = []
                for state_name, value in zip(self.buildings_states_actions[uid]['states'],
                                             self.buildings_states_actions[uid]['states'].values()):
                    if value == True:
                        if state_name not in action_dependent_state:
                            s.append(building.sim_results[state_name][self.time_step])
                        elif state_name in action_dependent_state:
                            s.append(0.0)

                self.state.append(np.array(s, dtype=np.float32))
                building.reset()

            self.state = np.array(self.state)

        return self._get_ob()

    def _get_ob(self):
        return self.state

    def _terminal(self):
        is_terminal = bool(self.time_step >= self.simulation_period[1] - self.simulation_period[0])
        if is_terminal:
            for building in self.buildings.values():
                building.terminate()

            # When the simulation is over, convert all the control variables to numpy arrays so they are easier to plot.
            self.net_electric_consumption = np.array(self.net_electric_consumption)
            self.net_electric_consumption_no_storage = np.array(self.net_electric_consumption_no_storage)
            self.net_electric_consumption_no_pv_no_storage = np.array(self.net_electric_consumption_no_pv_no_storage)
            self.electric_consumption_dhw_storage = np.array(self.electric_consumption_dhw_storage)
            self.electric_consumption_cooling_storage = np.array(self.electric_consumption_cooling_storage)
            self.electric_consumption_dhw = np.array(self.electric_consumption_dhw)
            self.electric_consumption_cooling = np.array(self.electric_consumption_cooling)
            self.electric_consumption_appliances = np.array(self.electric_consumption_appliances)
            self.electric_generation = np.array(self.electric_generation)
            self.district_power = np.array(self.district_power)
            self.loss.append([i for i in self.get_baseline_cost().values()])

            if self.verbose == 1:
                print('Cumulated reward: ' + str(self.cumulated_reward_episode))

        return is_terminal

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def cost(self):

        # Running the reference rule-based controller to find the baseline cost
        if self.cost_rbc is None:
            env_rbc = CityLearn_3dem(self.data_path, self.building_attributes, self.weather_file, self.solar_profile,
                                     self.el_data, self.building_ids,
                                     buildings_states_actions=self.buildings_states_actions_filename,
                                     building_dynamics_state=self.building_dynamics_state_filename,
                                     simulation_period=self.simulation_period, cost_function=self.cost_function,
                                     central_agent=False)
            _, actions_spaces = env_rbc.get_state_action_spaces()

            # Instantiatiing the control agent(s)
            agent_rbc = RBC_Agent(actions_spaces)

            state = env_rbc.reset()
            done = False
            while not done:
                action = agent_rbc.select_action(state)
                next_state, rewards, done, _= env_rbc.step(action)
                state = next_state
            self.cost_rbc = env_rbc.get_baseline_cost()

        # Compute the costs normalized by the baseline costs
        cost = {}
        if 'ramping' in self.cost_function:
            cost['ramping'] = np.abs(
                (self.net_electric_consumption - np.roll(self.net_electric_consumption, 1))[1:]).sum() / self.cost_rbc[
                                  'ramping']

        # Finds the load factor for every month (average monthly demand divided by its maximum peak), and averages all the load factors across the 12 months. The metric is one minus the load factor.
        if '1-load_factor' in self.cost_function:
            cost['1-load_factor'] = np.mean([1 - np.mean(self.net_electric_consumption[i:i + int(8760 / 12)]) / np.max(
                self.net_electric_consumption[i:i + int(8760 / 12)]) for i in
                                             range(0, len(self.net_electric_consumption), int(8760 / 12))]) / \
                                    self.cost_rbc['1-load_factor']

        # Average of all the daily peaks of the 365 day of the year. The peaks are calculated using the net energy demand of the whole district of buildings.
        if 'average_daily_peak' in self.cost_function:
            cost['average_daily_peak'] = np.mean([self.net_electric_consumption[i:i + 24].max() for i in
                                                  range(0, len(self.net_electric_consumption), 24)]) / self.cost_rbc[
                                             'average_daily_peak']

        # Peak demand of the district for the whole year period.
        if 'peak_demand' in self.cost_function:
            cost['peak_demand'] = self.net_electric_consumption.max() / self.cost_rbc['peak_demand']

        # Positive net electricity consumption for the whole district. It is clipped at a min. value of 0 because the objective is to minimize the energy consumed in the district, not to profit from the excess generation. (Island operation is therefore incentivized)
        if 'net_electricity_consumption' in self.cost_function:
            cost['net_electricity_consumption'] = self.net_electric_consumption.clip(min=0).sum() / self.cost_rbc[
                'net_electricity_consumption']

        # Not used for the challenge
        if 'quadratic' in self.cost_function:
            cost['quadratic'] = (self.net_electric_consumption.clip(min=0) ** 2).sum() / self.cost_rbc['quadratic']

        cost['total'] = np.mean([c for c in cost.values()])

        return cost

    def get_baseline_cost(self):

        # Computes the costs for the Rule-based controller, which are used to normalized the actual costs.
        cost = {}
        if 'ramping' in self.cost_function:
            cost['ramping'] = np.abs(
                (self.net_electric_consumption - np.roll(self.net_electric_consumption, 1))[1:]).sum()

        if '1-load_factor' in self.cost_function:
            cost['1-load_factor'] = np.mean([1 - np.mean(self.net_electric_consumption[i:i + int(8760 / 12)]) / np.max(
                self.net_electric_consumption[i:i + int(8760 / 12)]) for i in
                                             range(0, len(self.net_electric_consumption), int(8760 / 12))])

        if 'average_daily_peak' in self.cost_function:
            cost['average_daily_peak'] = np.mean([self.net_electric_consumption[i:i + 24].max() for i in
                                                  range(0, len(self.net_electric_consumption), 24)])

        if 'peak_demand' in self.cost_function:
            cost['peak_demand'] = self.net_electric_consumption.max()

        if 'net_electricity_consumption' in self.cost_function:
            cost['net_electricity_consumption'] = self.net_electric_consumption.clip(min=0).sum()

        if 'quadratic' in self.cost_function:
            cost['quadratic'] = (self.net_electric_consumption.clip(min=0) ** 2).sum()

        return cost