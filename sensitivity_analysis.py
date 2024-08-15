
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import os
from scipy.stats import kruskal

def main():
    # Create directory for sensitivity analysis results:
    os.makedirs("sensitivity_analysis", exist_ok=True)

    file = "mf_clam/mil_eval/00133_clam_sb_uni_macenko_ext_set/00000-clam_sb/predictions.csv"
    results = pd.read_csv(file)
    print(results)
    annotations_to_merge = "annotations_extra.csv"
    annotations = pd.read_csv(annotations_to_merge, sep=';')
    print(annotations)

    # Merge the dataframes column wise based on 'slide':
    merged = pd.merge(results, annotations, on='slide')

    # Only retain rows without NaN
    merged = merged.dropna()

    # Create a column called correct if the 'y_pred1' > 0.5 and 'y_true' == 1 or 'y_pred0' > 0.5 and 'y_true' == 0
    merged['y_pred'] = merged[['y_pred0', 'y_pred1']].idxmax(axis=1).apply(lambda x: int(x[-1]))
    merged['accuracy'] = (merged['y_pred'] == merged['y_true']).astype(int)
    print(merged)
    
    print("Overall accuracy")
    # Calculate the overall accuracy
    accuracy = merged['accuracy'].mean()
    balanced_accuracy = balanced_accuracy_score(merged['y_true'], merged['y_pred'])
    auc = roc_auc_score(merged['y_true'], merged['y_pred1'])
    print(f"Accuracy: {accuracy}")
    print(f"Balanced accuracy: {balanced_accuracy}")
    print(f"AUC: {auc}")

    # Initialize a list to store the test results
    test_results = []

    # Function to perform and print Kruskal-Wallis H-test and store results
    def perform_kruskal_test(data, group_col, value_col):
        groups = data[group_col].unique()
        samples = [data[data[group_col] == group][value_col] for group in groups]
        stat, p_value = kruskal(*samples)
        print(f'Kruskal-Wallis H-test for {group_col} on {value_col}: H-statistic = {stat}, p-value = {p_value}')
        test_results.append({'group': group_col, 'value': value_col, 'H-statistic': stat, 'p-value': p_value})
        return stat, p_value

    print("Check for difficulty:")
    # Calculate the accuracy for each group in 'difficulty'
    print("Accuracy:")
    accuracy = merged.groupby('difficulty')['accuracy'].mean()
    print(accuracy)
    # Calculate the uncertainty for each group in 'difficulty'
    print("Uncertainty:")
    uncertainty = merged.groupby('difficulty')['uncertainty'].mean()
    print(uncertainty)
    # Save accuracy and uncertainty for difficulty to CSV
    difficulty_results = pd.DataFrame({'difficulty': accuracy.index, 'accuracy': accuracy.values, 'uncertainty': uncertainty.values})
    difficulty_results.to_csv('sensitivity_analysis/difficulty_accuracy_uncertainty.csv', index=False)
    # Plot the distribution of uncertainty for each group in 'difficulty'
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=merged, x='difficulty', y='uncertainty', scale='width')
    plt.ylim(0, 1)
    plt.title("Uncertainty distribution for each difficulty")
    plt.xlabel("Difficulty")
    plt.ylabel("Uncertainty")
    plt.savefig('sensitivity_analysis/difficulty_uncertainty.png')
    plt.close()
    # Perform Kruskal-Wallis H-test for 'difficulty' on 'uncertainty'
    perform_kruskal_test(merged, 'difficulty', 'uncertainty')
    # Perform Kruskal-Wallis H-test for 'difficulty' on 'accuracy'
    perform_kruskal_test(merged, 'difficulty', 'accuracy')

    print("Check for biopsy_location:")
    # Calculate the accuracy for each group in 'biopsy_location'
    print("Accuracy:")
    accuracy = merged.groupby('biopsy_location')['accuracy'].mean()
    print(accuracy)
    # Calculate the uncertainty for each group in 'biopsy_location'
    print("Uncertainty:")
    uncertainty = merged.groupby('biopsy_location')['uncertainty'].mean()
    print(uncertainty)
    # Save accuracy and uncertainty for biopsy_location to CSV
    biopsy_location_results = pd.DataFrame({'biopsy_location': accuracy.index, 'accuracy': accuracy.values, 'uncertainty': uncertainty.values})
    biopsy_location_results.to_csv('sensitivity_analysis/biopsy_location_accuracy_uncertainty.csv', index=False)
    # Plot the distribution of uncertainty for each group in 'biopsy_location'
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=merged, x='biopsy_location', y='uncertainty', scale='width')
    plt.ylim(0, 1)
    plt.title("Uncertainty distribution for each biopsy location")
    plt.xlabel("Biopsy Location")
    plt.ylabel("Uncertainty")
    plt.savefig('sensitivity_analysis/biopsy_uncertainty.png')
    plt.close()
    # Perform Kruskal-Wallis H-test for 'biopsy_location' on 'uncertainty'
    perform_kruskal_test(merged, 'biopsy_location', 'uncertainty')
    # Perform Kruskal-Wallis H-test for 'biopsy_location' on 'accuracy'
    perform_kruskal_test(merged, 'biopsy_location', 'accuracy')

    print("Check for sex:")
    # Calculate the accuracy for each group in 'sex'
    accuracy = merged.groupby('sex')['accuracy'].mean()
    print(accuracy)
    # Calculate the uncertainty for each group in 'sex'
    uncertainty = merged.groupby('sex')['uncertainty'].mean()
    print(uncertainty)
    # Save accuracy and uncertainty for sex to CSV
    sex_results = pd.DataFrame({'sex': accuracy.index, 'accuracy': accuracy.values, 'uncertainty': uncertainty.values})
    sex_results.to_csv('sensitivity_analysis/sex_accuracy_uncertainty.csv', index=False)
    # Plot the distribution of uncertainty for each group in 'sex'
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=merged, x='sex', y='uncertainty', scale='width')
    plt.ylim(0, 1)
    plt.title("Uncertainty distribution for each sex")
    plt.xlabel("Sex")
    plt.ylabel("Uncertainty")
    plt.savefig('sensitivity_analysis/sex_uncertainty.png')
    plt.close()
    # Perform Kruskal-Wallis H-test for 'sex' on 'uncertainty'
    perform_kruskal_test(merged, 'sex', 'uncertainty')
    # Perform Kruskal-Wallis H-test for 'sex' on 'accuracy'
    perform_kruskal_test(merged, 'sex', 'accuracy')

    # Convert the test results to a DataFrame and save to CSV
    results_df = pd.DataFrame(test_results)
    results_df.to_csv('sensitivity_analysis/kruskal_wallis_test_results.csv', index=False)

if __name__ == "__main__":
    main()
