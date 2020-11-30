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
    p = [0]
    q = [0]
    
    for _, row in df.iterrows():
        if intervals[len(intervals)-1][1] >= row['Mhz']:
            m[len(m)-1] += int(row['m'])
            p[len(p)-1] += row['p']
        else:
            intervals.append(
                (intervals[len(intervals)-1][1], intervals[len(intervals)-1][1] + h)
            )
            s[len(s)-1] += m[len(m)-1]
            q[len(q)-1] += p[len(p)-1]
            s.append(s[len(s)-1])
            q.append(q[len(q)-1])

            m.append(int(row['m']))
            p.append(row['p'])

    s[len(s)-1] += m[len(m)-1]
    q[len(q)-1] += p[len(p)-1]
    
    interval_series_df = pd.DataFrame({
        'interval': intervals,
        'middles': list(map(lambda item: np.mean(item), intervals)),
        'm': m, 'p': p,
        's': s, 'q': q
    })
        
    print(interval_series_df)
    
    
if __name__ == "__main__":
    main()