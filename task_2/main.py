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

    interval_series_df = pd.DataFrame({
        'interval': intervals,
        'middles': middles,
        'm': m, 'p': p,
        's': s, 'q': q
    })
        
    print(interval_series_df, end='\n\n')
    
    C = middles[math.ceil(len(middles)/2)-1]
    z = list(map(lambda mid: int((mid - C) / h), middles))
    
    interval_series_df['con_vars'] = z
    print(interval_series_df, end='\n\n')

    M = list()
    for i in range(1, 5):
        M.append(sum(list(map(lambda m_i, z_i: (m_i * (z_i ** i)) / df_size, m, z))))

    μ_2 = M[1] - (M[0] ** 2)
    μ_3 = M[2] - (3 * M[1] * M[0]) + (2 * (M[0] ** 3))
    μ_4 = M[3] - (4 * M[2] * M[0]) + (6 * M[1] * (M[0] ** 2)) - (3 * (M[0] ** 4))
    μ = [0, μ_2, μ_3, μ_4]

    moments_df = pd.DataFrame({
        'M*': M,
        'μ*': μ
    })
    moments_df.index = moments_df.index + 1
    
    print(moments_df, end='\n\n')

    x_sample_mean = sum(list(map(lambda x, m: x * m, middles, m))) / df_size
    print(f"Sample mean: {round(x_sample_mean, 5)}")
    assert x_sample_mean == M[0] * h + C, "Error: the sample mean is not equal in two different formulas"

    D = sum(list(map(lambda x_i, m_i: ((x_i - x_sample_mean) ** 2) * m_i, middles, m))) / df_size

    print(f"Sample dispersion: {round(D, 5)}")   

    assert D == round(μ_2 * (h ** 2), 10), "Error: the sample dispersion is not equal in two different formulas"



    



    
if __name__ == "__main__":
    main()