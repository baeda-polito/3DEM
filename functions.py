import pandas as pd
import numpy as np

#KPI definition
#We can use a mask to introduce a ficticious selling electricity price if the electricity of the single building is smaller than 0
def economic_cost(env,building_ids):
    consumptions = pd.DataFrame()
    electricity = pd.DataFrame()
    single_costs = []
    tot_cost = []
    for uid in building_ids:
        consumptions = consumptions.append(pd.Series(env.buildings[uid].net_electric_consumption), ignore_index=True)
        electricity = electricity.append(pd.Series(env.buildings[uid].sim_results['el_price']),ignore_index=True)

    single_costs = (consumptions*electricity).sum(axis=1)
    tot_cost= single_costs.sum()
    return single_costs,tot_cost

def results(env,building_ids):
    results = pd.DataFrame(columns=['Cost', 'Peak', 'PAR', 'Daily Peak', 'Daily PAR', 'Flexibility Factor'])

    consumptions = pd.DataFrame()
    electricity = pd.DataFrame()
    single_costs = []
    for uid in building_ids:
        consumptions = consumptions.append(pd.Series(env.buildings[uid].net_electric_consumption), ignore_index=True)
        electricity = electricity.append(pd.Series(env.buildings[uid].sim_results['el_price']), ignore_index=True)

    single_costs = (consumptions * electricity).sum(axis=1)
    cost = single_costs.sum()
    peak = max(env.net_electric_consumption)
    par = max(env.net_electric_consumption) / np.mean(env.net_electric_consumption)

    daily_consumption = np.reshape(env.net_electric_consumption[11:], (-1, 24)).mean(axis=0)
    daily_peak = max(daily_consumption)
    daily_par = max(daily_consumption) / np.mean(daily_consumption)

    daily_cost = np.reshape(env.buildings[building_ids[0]].sim_results['el_price'][0:2196-12], (-1, 24))
    daily_cost = pd.DataFrame(daily_cost)
    mask = daily_cost == min(env.buildings[building_ids[0]].sim_results['el_price'][:])

    consumption_daily_24 = np.reshape(env.net_electric_consumption[11:], (-1, 24))
    consumption_daily_24 = pd.DataFrame(consumption_daily_24)
    en_off = consumption_daily_24[mask].fillna(0).sum().sum()
    flex_fac = en_off / (daily_consumption.sum().sum())

    results = results.append({'Cost': cost, 'Peak': peak, 'PAR': par, 'Daily Peak': daily_peak, 'Daily PAR': daily_par,
                              'Flexibility Factor': flex_fac}, ignore_index=True)

    return results


#-----------------------------------------comfort analysis

def discomfort(env,building_ids):

    discomfort = pd.DataFrame(columns=['hour_cold_disc','hour_hot_disc','degree_cold_disc','degree_hot_disc','average_cold_disc','average_hot_disc'])
    for uid in building_ids:
        Tlstm = env.buildings[uid].lstm_results['T_lstm']
        Set_point = env.buildings[uid].sim_results['set_point']
        hour_lower_discomfort = 0
        hour_higher_discomfort = 0
        degree_lower_discomfort = 0
        degree_higher_discomfort = 0
        for i in range(len(Tlstm)):
            if Set_point[i] == 26:
                if Tlstm[i] < 25:
                    hour_lower_discomfort += 1
                    degree_lower_discomfort += abs(Tlstm[i] - 25)
                elif Tlstm[i] > 27:
                    hour_higher_discomfort += 1
                    degree_higher_discomfort += abs(Tlstm[i] - 27)

        if hour_lower_discomfort > 0:
            average_cold_discomfort = degree_lower_discomfort/hour_lower_discomfort
        else:
            average_cold_discomfort = 0

        if hour_higher_discomfort > 0:
            average_hot_discomfort = degree_higher_discomfort/hour_higher_discomfort
        else:
            average_hot_discomfort = 0

        discomfort = discomfort.append({'hour_cold_disc':hour_lower_discomfort,'hour_hot_disc':hour_higher_discomfort,
                           'degree_cold_disc':degree_lower_discomfort,'degree_hot_disc':degree_higher_discomfort,
                                       'average_cold_disc':average_cold_discomfort,'average_hot_disc':average_hot_discomfort}, ignore_index=True)


    return discomfort

