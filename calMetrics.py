import pandas as pd
import argparse
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--document", type=str, default='dataset11')
parser.add_argument("--rand", type=int, default=0)
args = parser.parse_args()

Data =args.document 
outdir = 'result/' + Data + '/'
if not os.path.exists(outdir):
    os.mkdir(outdir)

PATH = 'Result/'
impute_count_dir = PATH + Data
impute_count = os.listdir(impute_count_dir)
impute_count = [x for x in impute_count if x [-3:] == 'csv']


def calculate_scores(prefix):
    scores_df = pd.read_table(prefix+'_gene_Metrics.txt',index_col=0)
    scores_df['Rank_PCC'] = scores_df['PCC'].rank(ascending=False, method='min')
    scores_df['Rank_SSIM'] = scores_df['SSIM'].rank(ascending=False, method='min')
    scores_df['Rank_RMSE'] = scores_df['RMSE'].rank(ascending=True, method='min')
    scores_df['Rank_JS'] = scores_df['JS'].rank(ascending=True, method='min')
    scores_df['AS'] = (scores_df['Rank_PCC'] + scores_df['Rank_SSIM'] + scores_df['Rank_RMSE'] + scores_df['Rank_JS']) / 4 / (scores_df.shape[0]-1)
    return scores_df


def plot_metric(data_as, methods, metric_name, data_path, Data):
    save_path = data_path + '/AS/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    colors = ['#1f77b4', '#ebb1c9', '#e1d64c', '#c0c738', '#82b679', '#bcadcd', '#a8c6c9', '#8bcae3', '#9b95bc']

    fig, ax = plt.subplots(figsize=(10, 10))
    bp = ax.boxplot([data_as[method] for method in methods if method in data_as], patch_artist=True, notch=False,
                    labels=methods,
                    boxprops=dict(linewidth=0.1, facecolor='none'),
                    medianprops={'color': 'black', 'linewidth': 0.2},
                    whiskerprops={'linewidth': 0.2},
                    capprops={'linewidth': 0.2}
                    )

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    for spine in ax.spines.values():
        spine.set_linewidth(0.2)

    ax.xaxis.set_tick_params(width=0.2)
    ax.yaxis.set_tick_params(width=0.2)

    ax.set_title(f'{Data} - {metric_name}')
    ax.set_ylabel(metric_name)
    ax.set_xlabel('Method')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(save_path, f'{Data}_{metric_name}.pdf'))
    plt.show()
    plt.close()

def plotResult():
    data_path = 'Result/' + Data
    save_path = data_path + '/AS/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    methods = []
    for impute_count_file in impute_count:
        method = impute_count_file.split('_')[0]
        methods.append(method)

    file_paths = [f"{data_path}/{method}_gene_result.txt" for method in methods]

    metrics = ['AS', 'PCC', 'SSIM', 'RMSE', 'JS']
    data_metrics = {metric: {} for metric in metrics}

    for method, file_path in zip(methods, file_paths):
        try:
            data = pd.read_table(file_path, sep='\t', index_col=0, header=0)
            for metric in metrics:
                if metric in data.columns:
                    if data[metric].dtype.kind not in 'biufc':
                        print(f"The {metric} column of {method} contains non - numerical data")
                    data_metrics[metric][method] = data[metric].dropna()
        except Exception as e:
            print(f"Unable to load data for {method}: {e}")

    for metric in metrics:
        plot_metric(data_metrics[metric], methods, metric, data_path, Data)


def cal():

    results = {}
    if len(impute_count)!=0:
        for impute_count_file in impute_count:
            method = impute_count_file.split('_')[0]
            prefix = impute_count_dir + '/' + method
            impute_count_file = impute_count_dir + '/' + impute_count_file
            print(impute_count_file)
            results[method] = calculate_scores(prefix)
            save_path = prefix+"_gene_result.txt"
            results[method].to_csv(save_path,sep='\t', header=1, index=1)

def main():
    print ('We are calculating the : ' + Data + '\n')
    cal()
    plotResult()

if __name__ == "__main__":
    main()