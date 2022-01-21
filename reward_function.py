
# Reward function for the multi-agent (decentralized) agents
def reward_function_ma(electricity_demand):
    total_energy_demand = 0
    for e in electricity_demand:
        total_energy_demand += -e

    price = max(total_energy_demand * 0.01, 0)

    for i in range(len(electricity_demand)):
        electricity_demand[i] = min(price * electricity_demand[i], 0)

    return electricity_demand

# Reward function for the single-agent (centralized) agent
def reward_function_sa(electric_demand):
    total_energy_demand = 0
    penalty = 0

    peak = 0
    for e in electric_demand:
        peak += (e ** 2)
        total_energy_demand += -e
    pt = max(total_energy_demand, 0)
    peak = max(peak, 0)

    return -pt-peak/1000