from scipy.stats.distributions import chi2
from scipy.stats import norm
import pandas as pd
import numpy as np
import math


def main():
    df = pd.read_csv('../task_1/sample.csv')
    df_size = len(df)
    
    dict_frequencies = df['Processor_MHz'].value_counts().to_dict()
    dict_frequencies = dict(sorted(dict_frequencies.items()))
    
    processor_mhz = list(dict_frequencies.keys())
    abs_frequencies = list(dict_frequencies.values())
    rel_frequencies = list(map(lambda x: round(x / df_size, 4), abs_frequencies))
    
    df = pd.DataFrame({
        'Mhz': processor_mhz,
        'm': abs_frequencies,
        'p': rel_frequencies
    })
    
    k = 1 + math.floor(math.log2(df_size))
    h = math.ceil((processor_mhz[-1] - processor_mhz[0]) / k)
    
    intervals = [(processor_mhz[0], processor_mhz[0] + h)]
    m = [0]
    s = [0]
    p = list()
    q = list()

    for mhz, absolute in zip(processor_mhz, abs_frequencies):
        if intervals[len(intervals)-1][1] >= mhz:
            m[len(m)-1] += absolute
        else:
            intervals.append(
                (intervals[len(intervals)-1][1], intervals[len(intervals)-1][1] + h)
            )
            s[len(s)-1] += m[len(m)-1]
            p.append(round(m[len(m)-1] / df_size,4))
            q.append(round(s[len(s)-1] / df_size,4))
            
            s.append(s[len(s)-1])
            m.append(absolute)

    s[len(s)-1] += m[len(m)-1]
    p.append(round(m[len(m)-1] / df_size,4))
    q.append(round(s[len(s)-1] / df_size,4))
    middles = list(map(lambda item: np.mean(item), intervals))
    
    C = middles[math.ceil(len(middles)/2)-1]
    z = list(map(lambda mid: int((mid - C) / h), middles))

    M = list()
    for i in range(1, 5):
        M.append(sum(list(map(lambda m_i, z_i: (m_i * (z_i ** i)) / df_size, m, z))))

    μ_2 = (M[1] - (M[0] ** 2)) * (h ** 2)
    μ_3 = (M[2] - (3 * M[1] * M[0]) + (2 * (M[0] ** 3))) * (h ** 3)
    μ_4 = M[3] - ((4 * M[2] * M[0]) + (6 * M[1] * (M[0] ** 2)) - (3 * (M[0] ** 4))) * (h ** 4)

    x_sample_mean = M[0] * h + C    
    D = μ_2
    σ = math.sqrt(D)
    S2 = (df_size / (df_size - 1)) * D
    𝛾 = [0.95, 0.99]
    t_95 = 1.984984311431769
    t_99 = 2.628015842465222
    t = [round(t_95, 4), round(t_99, 4)]

    print(f'Corriected varience: {round(S2, 4)}', end='\n\n')
    ε_95 = (t_95 * math.sqrt(S2))/math.sqrt(df_size)
    ε_99 = (t_99 * math.sqrt(S2))/math.sqrt(df_size)
    ε_math = [round(ε_95, 4), round(ε_99, 4)]

    conf_interval_math = [
        (round(x_sample_mean-ε_95, 4), round(x_sample_mean+ε_95, 4)),
        (round(x_sample_mean-ε_99, 4), round(x_sample_mean+ε_99, 4))
    ]

    df_conf_intervals = pd.DataFrame({
        '𝛾': 𝛾, 't': t,
        'ε': ε_math, 'conf_interval_math': conf_interval_math
    })

    print(df_conf_intervals, end='\n\n')

    q_95 = round(math.sqrt((df_size-1)/chi2.ppf(0.05, df_size-1)) - 1, 4)
    q_99 = round(math.sqrt((df_size-1)/chi2.ppf(0.01, df_size-1)) - 1, 4)
    σ_fixed = math.sqrt(S2)
    
    conf_interval_sqo = [
        (round(σ_fixed*(1-q_95), 4), round(σ_fixed*(1+q_95), 4)),
        (round(σ_fixed*(1-q_99), 4), round(σ_fixed*(1+q_99), 4))
    ]
    
    df_conf_intervals_sqo = pd.DataFrame({
        '𝛾': 𝛾, 'q': [q_95, q_99], 'conf_interval_sqo': conf_interval_sqo
    })
    
    print(df_conf_intervals_sqo, end='\n\n')

    theoretic_freq = list()    
    for item in intervals:
        p_i = norm.cdf((item[1] - x_sample_mean)/σ) - norm.cdf((item[0] - x_sample_mean)/σ)    
        theoretic_freq.append(df_size * p_i)
    
    df_theoretic_freq = pd.DataFrame({
        'Interval': intervals,
        'm': m,
        'm`': theoretic_freq
    })

    print(df_theoretic_freq, end='\n\n')

    Xn = sum(list(map(lambda m_i, m_theor: ((m_i - m_theor) ** 2) / m_theor, m, theoretic_freq)))

    print(f'Observe Chi: {Xn}')
    
    Xcritical = chi2.ppf(0.95, k - 2 - 1)
    print(f'Chi-squad critical: {Xcritical}')


    
if __name__ == "__main__":
    main()
    