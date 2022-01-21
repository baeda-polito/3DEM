from gym import spaces
import numpy as np
import torch

class Building:
    def __init__(self, buildingId, dhw_storage=None, cooling_storage=None, electrical_storage=None,
                 dhw_heating_device=None, cooling_device=None , model_dynamics = None):
        """
        Args:
            buildingId (int)
            dhw_storage (EnergyStorage)
            cooling_storage (EnergyStorage)
            electrical_storage (EnergyStorage)
            dhw_heating_device (ElectricHeater or HeatPump)
            cooling_device (HeatPump)
            model_dynamics (BuildingDynamics)
        """

        # Building attributes
        self.building_type = None
        self.climate_zone = None
        self.solar_power_capacity = None
        self.model_dynamics = model_dynamics
        self.buildingId = buildingId
        self.dhw_storage = dhw_storage
        self.cooling_storage = cooling_storage
        self.electrical_storage = electrical_storage
        self.dhw_heating_device = dhw_heating_device
        self.cooling_device = cooling_device
        self.observation_space = None
        self.action_space = None
        self.time_step = 0
        self.sim_results = {}
        self.sim_results_lagged = {}
        self.lstm_results = {}
        self.reset()

    def set_state_space(self, high_state, low_state):
        # Setting the state space and the lower and upper bounds of each state-variable
        self.observation_space = spaces.Box(low=low_state, high=high_state, dtype=np.float32)

    def set_action_space(self, max_action, min_action):
        # Setting the action space and the lower and upper bounds of each action-variable
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)



    def set_storage_heating(self, action):
        """
        Args:
            action (float): Amount of heating energy stored (added) in that time-step as a ratio of the maximum capacity of the energy storage device.
            -1 =< action < 0 : Energy Storage Unit releases energy into the building and its State of Charge decreases
            0 < action <= 1 : Energy Storage Unit receives energy from the energy supply device and its State of Charge increases
            The actions are always subject to the constraints of the power capacity of the heating supply unit, the DHW demand of the
            building (which limits the maximum amount of DHW that the energy storage can provide to the building), and the state of charge of the
            energy storage unit itself
        Return:
            elec_demand_heating (float): electricity consumption needed for space heating and heating storage
        """

        # Heating power that could be possible to supply to the storage device to increase its State of Charge once the heating demand of the building has been satisfied
        heat_power_avail = self.dhw_heating_device.get_max_heating_power() - self.sim_results['dhw_demand'][
            self.time_step]

        # The storage device is charged (action > 0) or discharged (action < 0) taking into account the max power available and that the storage device cannot be discharged by an amount of energy greater than the energy demand of the building.
        heating_energy_balance = self.dhw_storage.charge(max(-self.sim_results['dhw_demand'][self.time_step],
                                                             min(heat_power_avail, action * self.dhw_storage.capacity)))
        self.dhw_heating_device_to_storage.append(max(0, heating_energy_balance))
        self.dhw_storage_to_building.append(-min(0, heating_energy_balance))
        self.dhw_heating_device_to_building.append(
            self.sim_results['dhw_demand'][self.time_step] + min(0, heating_energy_balance))
        self.dhw_storage_soc.append(self.dhw_storage._soc)

        # The energy that the energy supply device must provide is the sum of the energy balance of the storage unit (how much net energy it will lose or get) plus the energy supplied to the building. A constraint is added to guarantee it's always positive.
        heating_energy_balance = max(0, heating_energy_balance + self.sim_results['dhw_demand'][self.time_step])

        # Electricity consumed by the energy supply unit
        elec_demand_heating = self.dhw_heating_device.set_total_electric_consumption_heating(
            heat_supply=heating_energy_balance)
        self.electric_consumption_dhw.append(elec_demand_heating)

        # Electricity consumption used (if +) or saved (if -) due to the change in the state of charge of the energy storage device
        self._electric_consumption_dhw_storage = elec_demand_heating - self.dhw_heating_device.get_electric_consumption_heating(
            heat_supply=self.sim_results['dhw_demand'][self.time_step])
        self.electric_consumption_dhw_storage.append(self._electric_consumption_dhw_storage)

        self.dhw_heating_device.time_step += 1

        return elec_demand_heating

    def set_storage_cooling(self, action):
        """
            Args:
                action (float): Amount of cooling energy stored (added) in that time-step as a ratio of the maximum capacity of the energy storage device.
                1 =< action < 0 : Energy Storage Unit releases energy into the building and its State of Charge decreases
                0 < action <= -1 : Energy Storage Unit receives energy from the energy supply device and its State of Charge increases
                The actions are always subject to the constraints of the power capacity of the cooling supply unit, the cooling demand of the
                building (which limits the maximum amount of cooling energy that the energy storage can provide to the building), and the state of charge of the energy storage unit itself
            Return:
                elec_demand_cooling (float): electricity consumption needed for space cooling and cooling storage
        """
        # The heat pump capacity is evaluated according to the external temperature
        self.cooling_device.capacity_limited = \
        cop_curve_cooling(self.sim_results['cooling_demand'][self.time_step], self.sim_results['t_out'][self.time_step],
                          self.cooling_device.nominal_power, self.cooling_device.nominal_COP)[1]
        # Cooling power that could be possible to supply to the storage device to increase its State of Charge once the heating demand of the building has been satisfied
        cooling_power_avail = self.cooling_device.capacity_limited - self.sim_results['cooling_demand'][self.time_step]

        # The storage device is charged (action > 0) or discharged (action < 0) taking into account the max power available and that the storage device cannot be discharged by an amount of energy greater than the energy demand of the building.
        cooling_energy_balance = self.cooling_storage.charge(max(-self.sim_results['cooling_demand'][self.time_step],
                                                                 min(cooling_power_avail,
                                                                     action * self.cooling_storage.capacity)))
        self.cooling_device_to_storage.append(max(0, cooling_energy_balance))


        self.cooling_storage_to_building.append(-min(0, cooling_energy_balance))

        self.cooling_device_to_building.append(
            self.sim_results['cooling_demand'][self.time_step] + min(0, cooling_energy_balance))

        self.cooling_storage_soc.append(self.cooling_storage._soc)

        # The energy that the energy supply device must provide is the sum of the energy balance of the storage unit (how much net energy it will lose or get) plus the energy supplied to the building. A constraint is added to guarantee it's always positive.
        cooling_energy_balance = max(0, cooling_energy_balance + self.sim_results['cooling_demand'][self.time_step])

        # Electricity consumed by the energy supply unit
        elec_demand_cooling, cop_cooling, capacity = self.cooling_device.set_total_electric_consumption_cooling(
            cooling_supply=cooling_energy_balance, temperature=self.sim_results['t_out'][self.time_step])
        self.electric_consumption_cooling.append(elec_demand_cooling)
        self.cooling_device.cop_cooling.append(cop_cooling)
        self.cooling_device.capacity.append(capacity)
        # Electricity consumption used (if +) or saved (if -) due to the change in the state of charge of the energy storage device
        self._electric_consumption_cooling_storage = elec_demand_cooling - self.cooling_device.get_electric_consumption_cooling(
            cooling_supply=self.sim_results['cooling_demand'][self.time_step],
            temperature=self.sim_results['t_out'][self.time_step])
        self.electric_consumption_cooling_storage.append(self._electric_consumption_cooling_storage)

        self.cooling_device.time_step += 1
        self.cooling_device.cooling_power_available.append(cooling_power_avail)
        return elec_demand_cooling

    def get_non_shiftable_load(self):
        return self.sim_results['non_shiftable_load'][self.time_step]

    def get_solar_power(self):
        return self.sim_results['solar_gen'][self.time_step]

    def get_dhw_electric_demand(self):
        return self.dhw_heating_device._electrical_consumption_heating

    def get_cooling_electric_demand(self):
        return self.cooling_device._electrical_consumption_cooling


    def reset(self):
        if self.dhw_storage is not None:
            self.dhw_storage.reset()
        if self.cooling_storage is not None:
            self.cooling_storage.reset()
        if self.electrical_storage is not None:
            self.electrical_storage.reset()
        if self.dhw_heating_device is not None:
            self.dhw_heating_device.reset()
        if self.cooling_device is not None:
            self.cooling_device.reset()
        if self.model_dynamics is not None:
            self.model_dynamics.reset()

        self._electric_consumption_cooling_storage = 0.0
        self._electric_consumption_dhw_storage = 0.0

        self.cooling_demand_building = []
        self.dhw_demand_building = []
        self.electric_consumption_appliances = []
        self.electric_generation = []

        self.electric_consumption_cooling = []
        self.electric_consumption_cooling_storage = []
        self.electric_consumption_dhw = []
        self.electric_consumption_dhw_storage = []

        self.net_electric_consumption = []
        self.net_electric_consumption_no_storage = []
        self.net_electric_consumption_no_pv_no_storage = []

        self.cooling_device_to_building = []
        self.cooling_storage_to_building = []
        self.cooling_device_to_storage = []
        self.cooling_storage_soc = []

        self.dhw_heating_device_to_building = []
        self.dhw_storage_to_building = []
        self.dhw_heating_device_to_storage = []
        self.dhw_storage_soc = []

    def terminate(self):

        if self.dhw_storage is not None:
            self.dhw_storage.terminate()
        if self.cooling_storage is not None:
            self.cooling_storage.terminate()
        if self.electrical_storage is not None:
            self.electrical_storage.terminate()
        if self.dhw_heating_device is not None:
            self.dhw_heating_device.terminate()
        if self.cooling_device is not None:
            self.cooling_device.terminate()
        if self.model_dynamics is not None:
            self.model_dynamics.terminate()

        self.cooling_demand_building = np.array(self.sim_results['cooling_demand'][:self.time_step])
        self.dhw_demand_building = np.array(self.sim_results['dhw_demand'][:self.time_step])
        self.electric_consumption_appliances = np.array(self.sim_results['non_shiftable_load'][:self.time_step])
        self.electric_generation = np.array(self.sim_results['solar_gen'][:self.time_step])

        elec_consumption_dhw = 0
        elec_consumption_dhw_storage = 0
        if self.dhw_heating_device.time_step == self.time_step and self.dhw_heating_device is not None:
            elec_consumption_dhw = np.array(self.electric_consumption_dhw)
            elec_consumption_dhw_storage = np.array(self.electric_consumption_dhw_storage)

        elec_consumption_cooling = 0
        elec_consumption_cooling_storage = 0
        if self.cooling_device.time_step == self.time_step and self.cooling_device is not None:
            elec_consumption_cooling = np.array(self.electric_consumption_cooling)
            elec_consumption_cooling_storage = np.array(self.electric_consumption_cooling_storage)

        self.net_electric_consumption = np.array(
            self.electric_consumption_appliances) + elec_consumption_cooling + elec_consumption_dhw - np.array(
            self.electric_generation)
        self.net_electric_consumption_no_storage = np.array(self.electric_consumption_appliances) + (
                    elec_consumption_cooling - elec_consumption_cooling_storage) + (
                                                               elec_consumption_dhw - elec_consumption_dhw_storage) - np.array(
            self.electric_generation)
        self.net_electric_consumption_no_pv_no_storage = np.array(self.net_electric_consumption_no_storage) + np.array(
            self.electric_generation)

        self.cooling_demand_building = np.array(self.cooling_demand_building)
        self.dhw_demand_building = np.array(self.dhw_demand_building)
        self.electric_consumption_appliances = np.array(self.electric_consumption_appliances)
        self.electric_generation = np.array(self.electric_generation)

        self.electric_consumption_cooling = np.array(self.electric_consumption_cooling)
        self.electric_consumption_cooling_storage = np.array(self.electric_consumption_cooling_storage)
        self.electric_consumption_dhw = np.array(self.electric_consumption_dhw)
        self.electric_consumption_dhw_storage = np.array(self.electric_consumption_dhw_storage)

        self.net_electric_consumption = np.array(self.net_electric_consumption)
        self.net_electric_consumption_no_storage = np.array(self.net_electric_consumption_no_storage)
        self.net_electric_consumption_no_pv_no_storage = np.array(self.net_electric_consumption_no_pv_no_storage)

        self.cooling_device_to_building = np.array(self.cooling_device_to_building)
        self.cooling_storage_to_building = np.array(self.cooling_storage_to_building)
        self.cooling_device_to_storage = np.array(self.cooling_device_to_storage)
        self.cooling_storage_soc = np.array(self.cooling_storage_soc)

        self.dhw_heating_device_to_building = np.array(self.dhw_heating_device_to_building)
        self.dhw_storage_to_building = np.array(self.dhw_storage_to_building)
        self.dhw_heating_device_to_storage = np.array(self.dhw_heating_device_to_storage)
        self.dhw_storage_soc = np.array(self.dhw_storage_soc)

class BuildingDynamics(torch.nn.Module):
    def __init__(self, n_features, lookback, n_hidden, n_layers):
        super(BuildingDynamics, self).__init__()
        self.n_features = n_features
        self.lookback = lookback
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.l_lstm = torch.nn.LSTM(input_size=n_features,
                                    hidden_size=n_hidden,
                                    num_layers=n_layers,
                                    batch_first=True)
        self.l_linear = torch.nn.Linear(self.n_hidden, 1)

    def forward(self, x, h):
        batch_size, lookback, _ = x.size()
        lstm_out, h = self.l_lstm(x, h)
        out = lstm_out[:, -1, :]
        out_linear_transf = self.l_linear(out)
        return out_linear_transf, h

    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        hidden = (hidden_state, cell_state)
        return hidden

    def reset(self):
        return

    def terminate(self):
        return


class HeatPump:
    def __init__(self, nominal_power=None, eta_tech=None, t_target_heating=None, t_target_cooling=None):
        """
        Args:
            nominal_power (float): Maximum amount of electric power that the heat pump can consume from the power grid (given by the nominal power of the compressor)
            eta_tech (float): Technical efficiency
            t_target_heating (float): Temperature at which the heating energy is released
            t_target_cooling (float): Temperature at which the cooling energy is released
        """
        # Parameters
        self.nominal_power = nominal_power
        self.eta_tech = eta_tech
        self.nominal_COP = 2.9
        # Variables
        self.max_cooling = None
        self.max_heating = None
        self._cop_heating = None
        self._cop_cooling = None
        self._capacity = None
        self.t_target_heating = t_target_heating
        self.t_target_cooling = t_target_cooling
        self.t_source_heating = None
        self.t_source_cooling = None
        self.cop_heating = []
        self.cop_cooling = []
        self.capacity = []
        self.electrical_consumption_cooling = []
        self.electrical_consumption_heating = []
        self.heat_supply = []
        self.cooling_supply = []
        self.time_step = 0
        self.cooling_power_available = []

    def get_max_cooling_power(self, max_electric_power=None):
        """
        Args:
            max_electric_power (float): Maximum amount of electric power that the heat pump can consume from the power grid

        Returns:
            max_cooling (float): maximum amount of cooling energy that the heatpump can provide
        """

        if max_electric_power is None:
            # self.max_cooling = self.nominal_power*self.cop_cooling[self.time_step] zolty
            self.max_cooling = self.nominal_power
        else:
            self.max_cooling = min(self.capacity[:self.time_step], self.nominal_power)
            # self.max_cooling = min(max_electric_power, self.nominal_power) * self.cop_cooling[self.time_step]
        return self.max_cooling

    def get_max_heating_power(self, max_electric_power=None):
        """
        Method that calculates the heating COP and the maximum heating power available
        Args:
            max_electric_power (float): Maximum amount of electric power that the heat pump can consume from the power grid

        Returns:
            max_heating (float): maximum amount of heating energy that the heatpump can provide
        """

        if max_electric_power is None:
            self.max_heating = self.nominal_power * self.cop_cooling[self.time_step]
        else:
            self.max_heating = min(max_electric_power, self.nominal_power) * self.cop_cooling[self.time_step]
        return self.max_heating

    def set_total_electric_consumption_cooling(self, cooling_supply=0, temperature=20):
        """
        Method that calculates the total electricity consumption of the heat pump given an amount of cooling energy to be supplied to both the building and the storage unit
        Args:
            cooling_supply (float): Total amount of cooling energy that the heat pump is going to supply

        Returns:
            _electrical_consumption_cooling (float): electricity consumption for cooling
            :param cooling_supply:
            :param temperature:
        """

        self.cooling_supply.append(cooling_supply)
        # to understand whether heat pump is functioning or not, cop is set to 0 if electrical consumption for cooling is 0
        if cooling_supply == 0:
            self._electrical_consumption_cooling = 0

            self._cop_cooling = 0
            self._capacity = cop_curve_cooling(cooling_supply, temperature, self.nominal_power, self.nominal_COP)[1]
        else:
            self._cop_cooling  = cop_curve_cooling(cooling_supply, temperature, self.nominal_power,
                                                                  self.nominal_COP)[0]
            self._capacity = cop_curve_cooling(cooling_supply, temperature, self.nominal_power,
                                                                  self.nominal_COP)[1]
            self._electrical_consumption_cooling = cooling_supply / self._cop_cooling

        self.electrical_consumption_cooling.append(self._electrical_consumption_cooling)
        return self._electrical_consumption_cooling, self._cop_cooling, self._capacity

    def get_electric_consumption_cooling(self, cooling_supply=0, temperature=0):
        """
        Method that calculates the electricity consumption of the heat pump given an amount of cooling energy
        Args:
            cooling_supply (float): Amount of cooling energy

        Returns:
            _electrical_consumption_cooling (float): electricity consumption for that amount of cooling
        """
        if cooling_supply == 0:
            _elec_consumption_cooling = 0
            # set cop to 0 if cooling is 0
            _cop_cooling = 0
            _capacity = cop_curve_cooling(cooling_supply, temperature, self.nominal_power, self.nominal_COP)[1]
        else:
            _cop_cooling = cop_curve_cooling(cooling_supply, temperature, self.nominal_power,
                                                        self.nominal_COP)[0]
            _capacity = cop_curve_cooling(cooling_supply, temperature, self.nominal_power,
                                                        self.nominal_COP)[1]
            _elec_consumption_cooling = cooling_supply / _cop_cooling
        return _elec_consumption_cooling

    def set_total_electric_consumption_heating(self, heat_supply=0):
        """
        Method that calculates the electricity consumption of the heat pump given an amount of heating energy to be supplied
        Args:
            heat_supply (float): Amount of heating energy that the heat pump is going to supply

        Returns:
            _elec_consumption_heating (float): electricity consumption for heating
        """

        self.heat_supply.append(heat_supply)
        self._electrical_consumption_heating = heat_supply / self.cop_heating[self.time_step]
        self.electrical_consumption_heating.append(self._electrical_consumption_heating)
        return self._electrical_consumption_heating

    def get_electric_consumption_heating(self, heat_supply=0):
        """
        Method that calculates the electricity consumption of the heat pump given an amount of heating energy to be supplied
        Args:
            heat_supply (float): Amount of heating energy that the heat pump is going to supply

        Returns:
            _elec_consumption_heating (float): electricity consumption for heating
        """

        _elec_consumption_heating = heat_supply / self.cop_heating[self.time_step]
        return _elec_consumption_heating

    def reset(self):
        self.t_source_heating = None
        self.t_source_cooling = None
        self.max_cooling = None
        self.max_heating = None
        self._cop_heating = None
        self._cop_cooling = None
        self._electrical_consumption_cooling = 0
        self._electrical_consumption_heating = 0
        self.electrical_consumption_cooling = []
        self.electrical_consumption_heating = []
        self.heat_supply = []
        self.cooling_supply = []
        self.time_step = 0
        self.cop_cooling = []

    def terminate(self):
        self.cop_heating = self.cop_heating[:self.time_step]
        self.cop_cooling = self.cop_cooling[:self.time_step]
        self.electrical_consumption_cooling = np.array(self.electrical_consumption_cooling)
        self.electrical_consumption_heating = np.array(self.electrical_consumption_heating)
        self.heat_supply = np.array(self.heat_supply)
        self.cooling_supply = np.array(self.cooling_supply)


class ElectricHeater:
    def __init__(self, nominal_power=None, efficiency=None):
        """
        Args:
            nominal_power (float): Maximum amount of electric power that the electric heater can consume from the power grid
            efficiency (float): efficiency
        """

        # Parameters
        self.nominal_power = nominal_power
        self.efficiency = efficiency

        # Variables
        self.max_heating = None
        self.electrical_consumption_heating = []
        self._electrical_consumption_heating = 0
        self.heat_supply = []
        self.time_step = 0

    def terminate(self):
        self.electrical_consumption_heating = np.array(self.electrical_consumption_heating)
        self.heat_supply = np.array(self.heat_supply)

    def get_max_heating_power(self, max_electric_power=None, t_source_heating=None, t_target_heating=None):
        """Method that calculates the maximum heating power available
        Args:
            max_electric_power (float): Maximum amount of electric power that the electric heater can consume from the power grid
            t_source_heating (float): Not used by the electric heater
            t_target_heating (float): Not used by electric heater

        Returns:
            max_heating (float): maximum amount of heating energy that the electric heater can provide
        """

        if max_electric_power is None:
            self.max_heating = self.nominal_power * self.efficiency
        else:
            self.max_heating = self.max_electric_power * self.efficiency

        return self.max_heating

    def set_total_electric_consumption_heating(self, heat_supply=0):
        """
        Args:
            heat_supply (float): Amount of heating energy that the electric heater is going to supply

        Returns:
            _electrical_consumption_heating (float): electricity consumption for heating
        """

        self.heat_supply.append(heat_supply)
        self._electrical_consumption_heating = heat_supply / self.efficiency
        self.electrical_consumption_heating.append(self._electrical_consumption_heating)
        return self._electrical_consumption_heating

    def get_electric_consumption_heating(self, heat_supply=0):
        """
        Args:
            heat_supply (float): Amount of heating energy that the electric heater is going to supply

        Returns:
            _electrical_consumption_heating (float): electricity consumption for heating
        """

        _electrical_consumption_heating = heat_supply / self.efficiency
        return _electrical_consumption_heating

    def reset(self):
        self.max_heating = None
        self.electrical_consumption_heating = []
        self.heat_supply = []


class EnergyStorage:
    def __init__(self, capacity=None, max_power_output=None, max_power_charging=None, efficiency=1, loss_coeff=0):
        """
        Generic energy storage class. It can be used as a chilled water storage tank or a DHW storage tank
        Args:
            capacity (float): Maximum amount of energy that the storage unit is able to store (kWh)
            max_power_output (float): Maximum amount of power that the storage unit can output (kW)
            max_power_charging (float): Maximum amount of power that the storage unit can use to charge (kW)
            efficiency (float): Efficiency factor of charging and discharging the storage unit (from 0 to 1)
            loss_coeff (float): Loss coefficient used to calculate the amount of energy lost every hour (from 0 to 1)
        """

        self.capacity = capacity
        self.max_power_output = max_power_output
        self.max_power_charging = max_power_charging
        self.efficiency = efficiency
        self.loss_coeff = loss_coeff
        self.soc = []
        self._soc = 0  # State of Charge
        self.energy_balance = []
        self._energy_balance = 0

    def terminate(self):
        self.energy_balance = np.array(self.energy_balance)
        self.soc = np.array(self.soc)

    def charge(self, energy):
        """Method that controls both the energy CHARGE and DISCHARGE of the energy storage device
        energy < 0 -> Discharge
        energy > 0 -> Charge
        Args:
            energy (float): Amount of energy stored in that time-step (Wh)
        Return:
            energy_balance (float):
        """

        # The initial State Of Charge (SOC) is the previous SOC minus the energy losses
        soc_init = self._soc * (1 - self.loss_coeff)

        # Charging
        if energy >= 0:
            if self.max_power_charging is not None:
                energy = min(energy, self.max_power_charging)
            self._soc = soc_init + energy * self.efficiency

        # Discharging
        else:
            if self.max_power_output is not None:
                energy = max(-self.max_power_output, energy)
            self._soc = max(0, soc_init + energy / self.efficiency)

        if self.capacity is not None:
            self._soc = min(self._soc, self.capacity)

        # Calculating the energy balance with its external environment (amount of energy taken from or relseased to the environment)

        # Charging
        if energy >= 0:
            self._energy_balance = (self._soc - soc_init) / self.efficiency

        # Discharging
        else:
            self._energy_balance = (self._soc - soc_init) * self.efficiency

        self.energy_balance.append(self._energy_balance)
        self.soc.append(self._soc)
        return self._energy_balance

    def reset(self):
        self.soc = []
        self._soc = 0  # State of charge
        self.energy_balance = []  # Positive for energy entering the storage
        self._energy_balance = 0
        self.time_step = 0

#the def return the COP as a function of external temperature and partial load
#moreover it also returns the heat pump capacity as function of external temperature
def cop_curve_cooling(cooling_supply, temperature, nominal_power, nominal_COP):
    cooling_supply = np.abs(cooling_supply)
    if temperature <= 20:
        capacity = nominal_power * (0.0021 * 20 ** 2 - 0.0751 * 20 + 1.0739)
        _cop = nominal_COP * (-0.0859 * 20 + 4.001) * (
                (cooling_supply / capacity) / (0.9 * (cooling_supply / capacity) + 0.1))
        _cop_max = nominal_COP * (-0.0859 * 20 + 4.001)
        _cop = min(_cop, _cop_max)
        _cop = max(_cop, 1)
        _copt = nominal_COP * (-0.0859 * 20 + 4.001)
        _copt = max(_copt, 1)
    elif temperature > 35:
        capacity = nominal_power * (0.0021 * 35 ** 2 - 0.0751 * 35 + 1.0739)
        _cop = nominal_COP * (-0.0859 * temperature + 4.001) * (
                (cooling_supply / capacity) / (0.9 * (cooling_supply / capacity) + 0.1))
        _cop = max(_cop, 1)
        _copt = nominal_COP * (-0.0859 * 35 + 4.001)
        _copt = max(_copt, 1)
    else:
        capacity = nominal_power * (0.0021 * temperature ** 2 - 0.0751 * temperature + 1.0739)
        _cop = nominal_COP * (-0.0859 * temperature + 4.001) * (
                (cooling_supply / capacity) / (0.9 * (cooling_supply / capacity) + 0.1))
        _cop = max(_cop, 1)
        _copt = nominal_COP * (-0.0859 * temperature + 4.001)
        _copt = max(_copt,1)
    return _cop, capacity

#the def return the COP as a function only of external temperature
def cop_T(temperature, nominal_COP):
    if temperature <= 20:
        _copt = nominal_COP * (-0.0859 * 20 + 4.001)
    elif temperature > 35:
        _copt = nominal_COP * (-0.0859 * temperature + 4.001)
    else:
        _copt =  nominal_COP * (-0.0859 * temperature + 4.001)
    _copt = max(_copt,1)
    return _copt

def get_building_cooling_loads(action):
    new_cooling = action
    return new_cooling
