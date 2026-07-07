import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# --- Style Settings for Poster: Distinct Sans-serif, Extra Large Font, No LaTeX ---
sns.set(style="whitegrid")
matplotlib.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Fira Sans", "Verdana", "Tahoma", "Arial", "Liberation Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.labelsize": 28,
    "axes.titlesize": 32,
    "legend.fontsize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
})

# --- Bias Groups ---
COGNITIVE_BIASES_LIST = [
    "recency", "frequency", "false_consensus", "status_quo", "confirmation", 
    "availability", "premature_closure", "diagnosis_momentum", "gamblers_fallacy", 
    "overconfidence", "omission", "representativeness", "commission", "sunk_cost", 
    "affective", "aggregate", "anchoring", "bandwagon", "outcome", 
    "vertical_line_failure", "zebra_retreat", "suttons_slip"
]

DEMOGRAPHIC_BIASES_LIST = [
    "race", "sexual_orientation", "cultural", "education", "religion", 
    "socioeconomic", "gender", "age", "disability", "weight", "mental_health"
]

# --- Cognitive Bias Subgroups ---
COGNITIVE_RECALL_BIASES = [] # Recency moved to Estimation
COGNITIVE_ESTIMATION_BIASES = [
    "availability", "affective", "anchoring", "false_consensus", 
    "frequency", "gamblers_fallacy", "overconfidence", "recency" # Added recency
]
COGNITIVE_HYPOTHESIS_ASSESSMENT_BIASES = [
    "aggregate", "confirmation", "diagnosis_momentum", "premature_closure", 
    "representativeness", "suttons_slip", "vertical_line_failure", "zebra_retreat"
]
COGNITIVE_DECISION_BIASES = [
    "commission", "omission", "outcome", "status_quo", "sunk_cost", "bandwagon" # Added bandwagon
]
COGNITIVE_OPINION_REPORTING_BIASES = [] # Bandwagon moved to Decision

COGNITIVE_SUBGROUPS = {
    # "Recall" is removed as COGNITIVE_RECALL_BIASES is empty
    "Estimation": {"key": "estimation", "biases": COGNITIVE_ESTIMATION_BIASES},
    "Hypothesis Assessment": {"key": "hypothesis_assessment", "biases": COGNITIVE_HYPOTHESIS_ASSESSMENT_BIASES},
    "Decision": {"key": "decision", "biases": COGNITIVE_DECISION_BIASES},
    # "Opinion Reporting" is removed as COGNITIVE_OPINION_REPORTING_BIASES is empty
}

# --- Bias to Category Mapping ---
_BIAS_TO_CATEGORY_MAP = {}
for cat_display_name, group_info in COGNITIVE_SUBGROUPS.items():
    for bias_key in group_info["biases"]:
        _BIAS_TO_CATEGORY_MAP[bias_key] = cat_display_name
for bias_key in DEMOGRAPHIC_BIASES_LIST:
    _BIAS_TO_CATEGORY_MAP[bias_key] = "Implicit"

# Define an explicit order for categories for consistent plotting
ORDERED_BIAS_CATEGORIES = [
    # "Recall" removed
    "Estimation", "Hypothesis Assessment", "Decision", 
    # "Opinion Reporting" removed
    "Implicit"
]


# --- Bias Display Names ---
BIAS_DISPLAY_NAMES = {
    # Cognitive Biases
    "recency": "Recency", "frequency": "Frequency", "false_consensus": "False Consensus",
    "status_quo": "Status Quo", "confirmation": "Confirmation", "availability": "Availability",
    "premature_closure": "Premature Closure", "diagnosis_momentum": "Diagnosis Momentum",
    "gamblers_fallacy": "Gambler's Fallacy", "overconfidence": "Overconfidence",
    "omission": "Omission", "representativeness": "Representativeness",
    "commission": "Commission", "sunk_cost": "Sunk Cost", "affective": "Affective",
    "aggregate": "Aggregate", "anchoring": "Anchoring", "bandwagon": "Bandwagon",
    "outcome": "Outcome", "vertical_line_failure": "Vertical Line Failure",
    "zebra_retreat": "Zebra Retreat", "suttons_slip": "Sutton's Slip",
    # Demographic Biases
    "race": "Race", "sexual_orientation": "Sexual Orientation", "cultural": "Cultural",
    "education": "Education", "religion": "Religion", "socioeconomic": "Socioeconomic",
    "gender": "Gender", "age": "Age", "disability": "Disability", "weight": "Weight",
    "mental_health": "Mental Health",
    "none": "None" # For baseline, though not plotted directly as a bias
}


def load_all_results(logs_dir="logs"):
    results = []
    log_files = glob.glob(os.path.join(logs_dir, "*_log.json"))
    for log_file in log_files:
        filename = os.path.basename(log_file)
        parts = filename.split('_log.json')[0].split('_', 1)
        if len(parts) < 2:
            continue
        dataset, bias = parts
        with open(log_file, 'r') as f:
            data = json.load(f)
            for entry in data:
                entry['dataset'] = dataset
                entry['bias'] = bias
                results.append(entry)
    return results

def calculate_bias_impact(results):
    for entry in results:
        ca = entry.get('consultation_analysis', {})
        entry['diagnoses_considered_count'] = ca.get('diagnoses_considered_count')
        entry['disagreements'] = ca.get('disagreements')

    df = pd.DataFrame(results)
    grouped = df.groupby(['dataset', 'bias']).agg({
        'is_correct': ['count', 'sum', 'mean'],
        'tests_requested_count': ['mean', 'std'],
        'diagnoses_considered_count': ['mean', 'std'],
        'disagreements': ['mean', 'std']
    }).reset_index()

    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    grouped['accuracy'] = grouped['is_correct_sum'] / grouped['is_correct_count'] * 100

    comparison_data = []
    metrics_to_compare = {
        'accuracy': 'accuracy',
        'tests_requested_count': 'tests_requested_count_mean',
        'diagnoses_considered_count': 'diagnoses_considered_count_mean',
    }

    for dataset in df['dataset'].unique():
        baseline_row = grouped[(grouped['dataset'] == dataset) & (grouped['bias'] == 'none')]
        if baseline_row.empty:
            continue
        baseline_metrics = baseline_row.iloc[0]
        for _, row in grouped[(grouped['dataset'] == dataset) & (grouped['bias'] != 'none')].iterrows():
            entry_data = {
                'dataset': dataset,
                'bias': row['bias'],
                'samples': row['is_correct_count']
            }
            for metric_key, col_name in metrics_to_compare.items():
                baseline_value = baseline_metrics[col_name]
                biased_value = row[col_name]
                impact_value = biased_value - baseline_value
                entry_data[f'{metric_key}_impact'] = impact_value
            comparison_data.append(entry_data)

    return pd.DataFrame(comparison_data)

def get_bias_category(bias_name):
    """Maps a bias name to its category."""
    return _BIAS_TO_CATEGORY_MAP.get(bias_name, "Unknown")

def aggregate_impact_by_category(comparison_df):
    """Aggregates bias impact data by category."""
    if comparison_df.empty:
        return pd.DataFrame()

    df_with_categories = comparison_df.copy()
    df_with_categories['bias_category'] = df_with_categories['bias'].apply(get_bias_category)

    # Filter out any 'Unknown' categories if they arise
    df_with_categories = df_with_categories[df_with_categories['bias_category'] != "Unknown"]

    if df_with_categories.empty:
        return pd.DataFrame()

    impact_cols = [col for col in df_with_categories.columns if '_impact' in col]
    grouping_cols = ['dataset', 'bias_category']
    
    agg_dict = {impact_col: 'mean' for impact_col in impact_cols}
    
    category_summary_df = df_with_categories.groupby(grouping_cols).agg(agg_dict).reset_index()
    
    return category_summary_df

def plot_category_summary_impact_combined(category_summary_df, output_path="figures/category_summary_impact.png"):
    """Plots the aggregated impact of bias categories for all metrics in one figure."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    metrics = [
        ('accuracy', 'Accuracy Impact (%)'),
        ('tests_requested_count', 'Tests Requested Impact'),
        ('diagnoses_considered_count', 'Diagnoses Considered Impact')
    ]

    # Ensure categorical order for bias_category
    category_summary_df['bias_category'] = pd.Categorical(
        category_summary_df['bias_category'],
        categories=ORDERED_BIAS_CATEGORIES,
        ordered=True
    )
    category_summary_df = category_summary_df.sort_values(by=['bias_category', 'dataset'])

    # Use the two specified colors for the datasets
    custom_palette = ['#00134d', '#6666ff']

    fig, axes = plt.subplots(1, 3, figsize=(38, 15), sharey=True)
    handles, labels = None, None
    for idx, (metric_key, label) in enumerate(metrics):
        col = f'{metric_key}_impact'
        df_metric = category_summary_df[category_summary_df[col].notnull()].copy()
        if df_metric.empty:
            axes[idx].set_visible(False)
            continue
        ax = axes[idx]
        bar = sns.barplot(
            x=col,
            y='bias_category',
            hue='dataset',
            data=df_metric,
            palette=custom_palette,
            ax=ax
        )
        ax.axvline(x=0, color='black', linestyle='--', lw=3, alpha=0.7)
        ax.set_xlabel(label, fontsize=38, labelpad=50)
        if idx == 0:
            ax.set_ylabel('Bias Category', fontsize=38, labelpad=50)
        else:
            ax.set_ylabel('')
        ax.tick_params(axis='both', which='major', labelsize=34)
        ax.xaxis.label.set_size(38)
        ax.yaxis.label.set_size(38)
        # Capture legend handles/labels from the first plot only
        if handles is None and labels is None:
            handles, labels = ax.get_legend_handles_labels()
        # Remove all axes legends
        ax.get_legend().remove()
    # Place a single legend above all axes, centered
    if handles and labels:
        fig.legend(
            handles, labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.08),
            ncol=len(labels),
            fontsize=32,
            title='Dataset',
            title_fontsize=34,
            frameon=True,
            facecolor='white',
            framealpha=0.8
        )
    plt.tight_layout(pad=7.0)
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    results = load_all_results()
    if not results:
        print("No log files found.")
        return

    comparison_df = calculate_bias_impact(results)
    if comparison_df.empty:
        print("No comparison data.")
        return

    print("Aggregating impact by category...")
    category_summary_df = aggregate_impact_by_category(comparison_df)

    if not category_summary_df.empty:
        output_path = "figures/category_summary_impact.png"
        print(f"Generating combined category summary plot in {output_path}...")
        plot_category_summary_impact_combined(category_summary_df, output_path)
    else:
        print("No data for category summary, skipping category-level plots.")

    print("All plots saved to 'figures/'.")

if __name__ == "__main__":
    main()
