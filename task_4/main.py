import pandas as pd
import numpy as np
import math


def main():
    df = pd.read_csv('../task_1/sample.csv')
    df_size = len(df)
    df = df.sort_values(by=['Processor_MHz'])

    dict_frequencies_mhz = df['Processor_MHz'].value_counts().to_dict()
    dict_frequencies_mhz = dict(sorted(dict_frequencies_mhz.items()))
    dict_frequencies_core = df['Cores'].value_counts().to_dict()
    dict_frequencies_core = dict(sorted(dict_frequencies_core.items()))

    processor_cores = list(dict_frequencies_core.keys())
    processor_mhz = list(dict_frequencies_mhz.keys())

    k = 1 + math.floor(math.log2(df_size))
    h_mhz = math.ceil((processor_mhz[-1] - processor_mhz[0]) / k)
    h_cores = math.ceil((processor_cores[-1] - processor_cores[0]) / k)
    
    intervals_mhz = [(processor_mhz[0], processor_mhz[0] + h_mhz)]
    intervals_cores = [(processor_cores[0], processor_cores[0] + h_cores)]

    for _ in range(k-1):
        intervals_mhz.append((intervals_mhz[len(intervals_mhz)-1][1], intervals_mhz[len(intervals_mhz)-1][1] + h_mhz))
        intervals_cores.append((intervals_cores[len(intervals_cores)-1][1], intervals_cores[len(intervals_cores)-1][1] + h_cores))

    middles_mhz = list(map(lambda item: np.mean(item), intervals_mhz))
    middles_cores = list(map(lambda item: np.mean(item), intervals_cores))

    m_binary = {
        intervals_mhz[i]: [0 for _ in range(7)] 
        for i in range(7)
    }
    
    
    for  _, row in df.iterrows():
            if (row['Processor_MHz'] >= intervals_mhz[0][0] and 
                row['Processor_MHz'] <= intervals_mhz[0][1] and
                row['Cores'] >= intervals_cores[0][0] and
                row['Cores'] <= intervals_cores[0][1]):
                m_binary[intervals_mhz[0]][0] += 1
    
    for i, mhz_item in zip(range(1, k), intervals_mhz[1:k+1]):
        for  _, row in df.iterrows():
                if (row['Processor_MHz'] > intervals_mhz[i][0] and 
                    row['Processor_MHz'] <= intervals_mhz[i][1] and
                    row['Cores'] >= intervals_cores[0][0] and
                    row['Cores'] <= intervals_cores[0][1]):
                    m_binary[mhz_item][0] += 1
                
                elif row['Processor_MHz'] > intervals_mhz[i][1]:
                    break
    
    for j, cores_item in zip(range(1, k), intervals_cores[1:k+1]):
        for  _, row in df.iterrows():
            if (row['Processor_MHz'] >= intervals_mhz[0][0] and 
                row['Processor_MHz'] <= intervals_mhz[0][1] and
                row['Cores'] > intervals_cores[j][0] and
                row['Cores'] <= intervals_cores[j][1]):
                m_binary[intervals_mhz[0]][j] += 1
    

    for i, mhz_item in zip(range(1, k), intervals_mhz[1:k+1]):
        for j in range(1, 7):
            for  _, row in df.iterrows():
                if (row['Processor_MHz'] > intervals_mhz[i][0] and 
                    row['Processor_MHz'] <= intervals_mhz[i][1] and
                    row['Cores'] > intervals_cores[j][0] and
                    row['Cores'] <= intervals_cores[j][1]):
                    m_binary[mhz_item][j] += 1
                
                elif row['Processor_MHz'] > intervals_mhz[i][1]:
                    break
    
    print("Binary interval series:")
    m_x = list()
    m_y = [0 for _ in range(k)]
    m = list()

    for _, value_item in m_binary.items():
        m_x.append(sum(value_item))
        m.append(value_item)
        for i in range(k):
            m_y[i] += value_item[i]
        
        print(value_item)
    print(end='\n\n')

    x_sample_mean = sum(list(map(lambda x, m: x * m, middles_mhz, m_x))) / df_size
    y_sample_mean = sum(list(map(lambda y, m: y * m, middles_cores, m_y))) / df_size
    
    D_x = sum(list(map(lambda x_i, m_i: ((x_i - x_sample_mean) ** 2) * m_i, middles_mhz, m_x))) / df_size
    D_y = sum(list(map(lambda x_i, m_i: ((x_i - y_sample_mean) ** 2) * m_i, middles_cores, m_y))) / df_size
    S2_x = (df_size / (df_size - 1)) * D_x
    S2_y = (df_size / (df_size - 1)) * D_y
    
    
    μ_в = 0
    for i in range(k):
        for j in range(k):
            μ_в += m[i][j] * (middles_mhz[i] - x_sample_mean) * (middles_cores[j] - y_sample_mean)
    μ_в /= 97
    print(μ_в)

    r_в = μ_в/(S2_x * S2_y)
    print(round((r_в), 7), end='\n\n')

    t_95 = 1.984984311431769
    t_99 = 2.628015842465222

    conf_interval_r = [
        (round(math.tanh(math.atanh(r_в) - (t_95/math.sqrt(df_size-3))), 5), 
        round(math.tanh(math.atanh(r_в) + (t_95/math.sqrt(df_size-3))), 5)),
        (round(math.tanh(math.atanh(r_в) - (t_99/math.sqrt(df_size-3))), 5),
         round(math.tanh(math.atanh(r_в) + (t_99/math.sqrt(df_size-3))), 5))
    ]

    conf_interval_p = [
        (
            (math.exp(2*conf_interval_r[0][0]) - 1)/(math.exp(2*conf_interval_r[0][0]) + 1),
            (math.exp(2*conf_interval_r[0][1]) - 1)/(math.exp(2*conf_interval_r[0][1]) + 1)
        ),
        (
            (math.exp(2*conf_interval_r[1][0]) - 1)/(math.exp(2*conf_interval_r[1][0]) + 1),
            (math.exp(2*conf_interval_r[1][1]) - 1)/(math.exp(2*conf_interval_r[1][1]) + 1)
        )
    ]
    

    print(conf_interval_p)

    t = r_в * math.sqrt(df_size - 2) / math.sqrt(1 - (r_в ** 2))
    print(t)

    
if __name__ == "__main__":
    main()
    