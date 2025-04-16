import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

root = '../datasets'
class_list = ['normal', 'preplus', 'plus']

def make_csv(csv_path):
    data = []
    for i, cls in enumerate(class_list):
        class_path = os.path.join(root, cls)
        for img in os.listdir(class_path):
            img_path = os.path.join(cls, img)
            data.append([img_path, i])
    
    data = np.array(data)
    df = pd.DataFrame(data, columns=['path', 'label'])
    
    # Add fold column using stratified k-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    df['fold'] = -1  # Initialize fold column
    
    # Get labels for stratification
    y = df['label'].values
    
    # Assign fold numbers
    for fold, (_, test_idx) in enumerate(skf.split(df, y)):
        df.loc[test_idx, 'fold'] = fold
    
    df.to_csv(csv_path, index=False)

def visualize_fold_distribution(csv_path):
    # Load the CSV file
    print("Loading CSV file...")
    df = pd.read_csv(csv_path)
    
    # Get class names for better visualization
    class_list = ['normal', 'preplus', 'plus']
    df['class_name'] = df['label'].apply(lambda x: class_list[x])
    
    # Count samples per class in each fold
    fold_distribution = pd.crosstab(df['fold'], df['class_name'])
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total samples: {len(df)}")
    print("\nClass distribution in the entire dataset:")
    class_counts = df['class_name'].value_counts()
    for cls, count in class_counts.items():
        print(f" {cls}: {count} samples ({count/len(df)*100:.2f}%)")
    
    print("\nClass distribution per fold:")
    for fold in range(5):
        fold_df = df[df['fold'] == fold]
        print(f"\nFold {fold} - Total: {len(fold_df)} samples")
        for cls in class_list:
            count = len(fold_df[fold_df['class_name'] == cls])
            print(f"  {cls}: {count} samples ({count/len(fold_df)*100:.2f}%)")
    
    # Create a grouped bar chart
    plt.figure(figsize=(10, 6))
    fold_distribution.plot(kind='bar', ax=plt.gca())
    plt.title('Class Distribution Across Folds', fontsize=16)
    plt.xlabel('Fold', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Class', fontsize=12)
    plt.tight_layout()
    # plt.savefig('../datasets/fold_distribution.png')
    
    # Create a stacked percentage bar chart
    plt.figure(figsize=(10, 6))
    fold_distribution_percent = fold_distribution.div(fold_distribution.sum(axis=1), axis=0) * 100
    fold_distribution_percent.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('Percentage Class Distribution Across Folds', fontsize=16)
    plt.xlabel('Fold', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Class', fontsize=12)
    for i in range(len(fold_distribution_percent)):
        total = fold_distribution.iloc[i].sum()
        plt.annotate(f'n={total}', 
                     xy=(i, 105), 
                     ha='center', 
                     fontsize=12)
    plt.tight_layout()
    # plt.savefig('../datasets/fold_distribution_percent.png')
    
    # Create a heatmap for visualization
    plt.figure(figsize=(10, 6))
    sns.heatmap(fold_distribution, annot=True, fmt='d', cmap='Blues')
    plt.title('Number of Samples per Class in Each Fold', fontsize=16)
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Fold', fontsize=14)
    plt.tight_layout()
    # plt.savefig('../datasets/fold_distribution_heatmap.png')
    
    print("\nVisualization completed. Images saved to '../datasets/' folder.")
    plt.show()

if __name__ == '__main__':
    # Define csv path
    csv_path = '../datasets/train_data_with_folds.csv'
    print("Creating csv file...")
    make_csv(csv_path)
    print("CSV file created successfully.")
#     print("Visualizing fold distribution...")
#     visualize_fold_distribution(csv_path)
