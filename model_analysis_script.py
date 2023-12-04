
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def extract_info_fp32(filename):
    '''Extract batch size and learning rate from fp32 filename'''
    parts = filename.split('_')
    batch_size = parts[2]
    learning_rate = parts[3].replace('.csv', '')
    return int(batch_size), float(learning_rate)

def extract_info_fp16(filename):
    '''Extract batch size and learning rate from fp16 filename'''
    parts = filename.split('_')
    batch_size = parts[3]
    learning_rate = parts[4].replace('.csv', '')
    return int(batch_size), float(learning_rate)

def read_and_aggregate_data(file_list, extraction_directory, precision_type):
    '''Read and aggregate data from given file list'''
    data = []
    for file in file_list:
        file_path = os.path.join(extraction_directory, file)
        df = pd.read_csv(file_path)

        if precision_type == 'fp32':
            batch_size, learning_rate = extract_info_fp32(file)
        else:
            batch_size, learning_rate = extract_info_fp16(file)

        df['batch_size'] = batch_size
        df['learning_rate'] = learning_rate

        aggregated_data = df.mean()
        aggregated_data['batch_size'] = batch_size
        aggregated_data['learning_rate'] = learning_rate

        data.append(aggregated_data)

    return pd.DataFrame(data)

def plot_3d_performance(data, title):
    '''Plot 3D performance graph'''
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    xs = data['batch_size']
    ys = data['learning_rate']
    zs = data['test_acc']

    ax.scatter(xs, ys, zs, color='blue', marker='o')

    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Learning Rate')
    ax.set_zlabel('Test Accuracy')
    ax.set_title(title)
    fig.savefig(os.path.join(os.getcwd(),'3d_performance.png'), dpi=220)
    plt.show()

def find_best_configurations(data):
    '''Find the best configurations for each metric'''
    best_configs = {}
    metrics = ['test_acc', 'test_precision', 'test_recall', 'test_f1_score', 'test_cohen_kappa', 'test_balanced_accuracy']

    for metric in metrics:
        best_row = data.loc[data[metric].idxmax()]
        best_configs[metric] = {
            'batch_size': best_row['batch_size'],
            'learning_rate': best_row['learning_rate'],
            metric: best_row[metric]
        }

    return best_configs

# Main script execution
if __name__ == '__main__':
    # Define the directory where the files are located
    extraction_directory = './test_results/'

    # Listing the contents of the extracted directory
    extracted_files = os.listdir(extraction_directory)

    # Sorting the files into fp32 and fp16 groups
    fp32_files = [f for f in extracted_files if not f.startswith('fp16')]
    fp16_files = [f for f in extracted_files if f.startswith('fp16')]

    # Processing fp32 and fp16 files
    fp32_data = read_and_aggregate_data(fp32_files, extraction_directory, 'fp32')
    fp16_data = read_and_aggregate_data(fp16_files, extraction_directory, 'fp16')

    # Plotting 3D graphs
    plot_3d_performance(fp32_data, 'FP32 Model Performance: Batch Size vs Learning Rate vs Accuracy')
    plot_3d_performance(fp16_data, 'FP16 Model Performance: Batch Size vs Learning Rate vs Accuracy')

    # Finding and printing best configurations
    best_configs_fp32 = find_best_configurations(fp32_data)
    best_configs_fp16 = find_best_configurations(fp16_data)
    print("Best Configurations for FP32:", best_configs_fp32)
    print("Best Configurations for FP16:", best_configs_fp16)
