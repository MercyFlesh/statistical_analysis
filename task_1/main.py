import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


def plot_graphs(intervals: list, m: list, s: list, name: str):
    middles_intervals = list(map(lambda item: np.mean(item), intervals))
    ends_intervals = list(map(lambda item: item[1], intervals))
    lengths_intervals = list(map(lambda item: item[1] - item[0], intervals))

    plt.figure()
    plt.plot(middles_intervals, m, marker='o')
    plt.grid()
    plt.xlabel('Mhz')
    plt.ylabel(f'{name} frequency')
    plt.title(f'Polygon {name} frequency')
    plt.savefig(f'polygon_{name}.png')
    
    plt.figure()
    plt.bar(middles_intervals, m, width=lengths_intervals, edgecolor='black')
    plt.grid()
    plt.xlabel('Mhz')
    plt.ylabel(f'{name} frequency')
    plt.title(f'Histogram {name} frequency')
    plt.savefig(f'histogram_{name}.png')

    plt.figure()
    plt.plot(ends_intervals, s, marker='o')
    plt.grid()
    plt.xlabel('Mhz')
    plt.ylabel(f'Accumulated {name} frequency')
    plt.title(f'Comulate {name} frequency')
    plt.savefig(f'comulate_{name}.png')

    plt.figure()
    plt.plot([0, middles_intervals[0]], [0, 0], color='blue')
    for i in range(0, len(middles_intervals)-1):
        plt.plot([middles_intervals[i], middles_intervals[i+1]], [s[i], s[i]], color='blue')
   
    plt.plot([middles_intervals[-1], middles_intervals[-1]+700], [s[-1], s[-1]], color='blue')
    plt.grid()
    plt.xlabel('Mhz')
    plt.ylabel(f'Accumulated {name} frequency')
    plt.title(f'Emperical distribution function of {name} frequency')
    plt.savefig(f'emperical_function_{name}.png')


def get_dataframe(csv_name: str, features: list, n: int):
    df = pd.read_csv(csv_name)
    df = df[features].sample(n)
    df.to_csv('sample.csv', index=False)
    return df


def main():
    df = get_dataframe('Gold.csv', ['Cores', 'Processor_MHz'], 97)
    df_size = len(df)
    
    dict_frequencies = df['Processor_MHz'].value_counts().to_dict()
    dict_frequencies = dict(sorted(dict_frequencies.items()))
    
    processor_mhz = list(dict_frequencies.keys())
    abs_frequencies = list(dict_frequencies.values())
    rel_frequencies = list(map(lambda x: round(x / df_size, 4), abs_frequencies))
    
    df = pd.DataFrame({
        'Processor_Mhz': processor_mhz,
        'm': abs_frequencies,
        'p': rel_frequencies
    })
    
    print(df, end='\n\n')
    
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
    
    interval_series_df = pd.DataFrame({
        'interval': intervals,
        'm': m, 'p': p,
        's': s, 'q': q
    })
        
    print(interval_series_df)

    plot_graphs(intervals, m, s, 'absolute')
    plot_graphs(intervals, p, q, 'relative')


if __name__ == "__main__":
    main()