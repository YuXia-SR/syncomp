import numpy as np
import tapas.datasets
import tapas.generators
import tapas.threat_models
import tapas.attacks
import tapas.report
import tempfile
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

def string_conv_int(x):
    mapping = {v: i for i, v in enumerate(set(x))}
    return np.array(list(map(mapping.__getitem__, x)))

def encode_integer(df):
    string_columns = df.select_dtypes(include=['object', 'bool']).columns

    # Encode string columns as integers
    for col in string_columns:
        df[col] = string_conv_int(df[col])

    return df

def compute_distance(df, row):
    distance = np.linalg.norm(df - row, axis=1)
    return distance.min()

def distance_closest_record(real_df, syn_df, n_sample=1000, random_state=0, n_clusters=10):
    real_df = real_df.sample(n_sample, random_state=random_state).fillna(0)
    syn_df = syn_df.fillna(0)

    real_df = encode_integer(real_df)
    syn_df = encode_integer(syn_df)

    kmeans_df1 = KMeans(n_clusters=n_clusters, random_state=random_state).fit(real_df)
    cluster_centers_df1 = kmeans_df1.cluster_centers_

    # Clustering df2
    kmeans_df2 = KMeans(n_clusters=n_clusters, random_state=random_state).fit(syn_df)
    cluster_centers_df2 = kmeans_df2.cluster_centers_

    min_distance = np.inf
    df1_index = 0
    df2_index = 0
    for i, center_df1 in enumerate(cluster_centers_df1):
        for j, center_df2 in enumerate(cluster_centers_df2):
            distance = np.linalg.norm(center_df1 - center_df2)
            if distance < min_distance:
                min_distance = distance
                df1_index = i
                df2_index = j

    # Select the cluster centers corresponding to the minimum distance
    df1_selected_cluster = real_df[kmeans_df1.labels_ == df1_index]
    df2_selected_cluster = syn_df[kmeans_df2.labels_ == df2_index]

    # Compute the distance between the selected cluster centers
    dcr = df1_selected_cluster.apply(lambda x: compute_distance(df2_selected_cluster, x), axis=1)

    return dcr.min()

def compute_distance_with_closest_cluster(row, compare_df, cluster_label, closest_cluster_index, row_index):
    selected_compare_df = compare_df[cluster_label.labels_ == closest_cluster_index[row_index]]
    distance = np.linalg.norm(selected_compare_df - row, axis=1)
    return distance.min()


def distance_closest_record_comparison(real_df, syn_df, holdout_df, sample_size=1000, random_state=0):
    syn_df = syn_df.dropna()
    sample_size = min(sample_size, syn_df.shape[0])
    real_df = encode_integer(real_df)
    syn_df = encode_integer(syn_df.sample(sample_size, random_state=random_state)).reset_index(drop=True)
    holdout_df = encode_integer(holdout_df)

    real_distance = syn_df.apply(lambda x: compute_distance(real_df, x), axis=1)
    holdout_distance = syn_df.apply(lambda x: compute_distance(holdout_df, x), axis=1)

    distance_comparison = pd.DataFrame({
        "distance_to_real": real_distance,
        "distance_to_holdout": holdout_distance
    })

    return distance_comparison

class GeneratorReadFromLocal(tapas.generators.Generator):
    def __init__(self, model="AutoDiff", random_state=0, dir="results"):
        self.model = model
        self.random_state = random_state
        self.dir = dir
        super().__init__()

    def fit(self, dataset):
        self.dataset = dataset
        self.trained = True

    def generate(self, num_samples, **kwargs):
        # load synthetic data df and schema json
        syn_df = pd.read_csv(f"{self.dir}/{self.model}/{self.random_state}/synthetic_data.csv")
        with open(f"{self.dir}/complete_dataset_filtered.json") as f:
            schema = json.load(f)
        # construct TabularDataset object
        with tempfile.TemporaryDirectory() as tmpdir:
            syn_df.to_csv(f"{tmpdir}/complete_dataset.csv", index=False)
            with open(f"{tmpdir}/complete_dataset.json", 'w') as f:
                json.dump(schema, f)
            syn_data = tapas.datasets.TabularDataset.read(f"{tmpdir}/complete_dataset")
        
        return syn_data.sample(num_samples, random_state=self.random_state)
    
    def __call__(self, dataset, num_samples, **kwargs):
        self.fit(dataset)
        return self.generate(num_samples, random_state = self.random_state)

def evaluate_tapas_attack(
    train_df: pd.DataFrame,
    model: str="AutoDiff",
    random_state: int=0,
    dir: str="results",
    n_sample: int=1000,
    auxiliary_split=0.5,
    num_training_records=5000,
    classifier=RandomForestClassifier(n_estimators=10),
):
    
    with open(f"{dir}/complete_dataset_filtered.json") as f:
        schema = json.load(f)
    with tempfile.TemporaryDirectory() as tmpdir:
        train_df.to_csv(f"{tmpdir}/complete_dataset.csv", index=False)
        with open(f"{tmpdir}/complete_dataset.json", 'w') as f:
            json.dump(schema, f)
        data = tapas.datasets.TabularDataset.read(f"{tmpdir}/complete_dataset")
    
    generator = GeneratorReadFromLocal(model=model, random_state=random_state, dir=dir)
    data_knowledge = tapas.threat_models.AuxiliaryDataKnowledge(
        data,
        auxiliary_split=auxiliary_split,
        num_training_records=num_training_records
    )
    sdg_knowledge = tapas.threat_models.BlackBoxKnowledge(
        generator,
        num_synthetic_records=num_training_records,
    )
    threat_model = tapas.threat_models.TargetedMIA(
        attacker_knowledge_data=data_knowledge,
        target_record=data.get_records([0]),
        attacker_knowledge_generator=sdg_knowledge,
        generate_pairs=True,
        replace_target=True
    )
    # attacker = tapas.attacks.ShadowModellingAttack(
    #     tapas.attacks.FeatureBasedSetClassifier(
    #         tapas.attacks.NaiveSetFeature() + tapas.attacks.HistSetFeature() + tapas.attacks.CorrSetFeature(),
    #         classifier
    #     ),
    #     label = "Groundhog"
    # )

    attacker = tapas.attacks.ClosestDistanceMIA(
        criterion=("threshold", 0), label="Direct Lookup"
    )
    attacker.train(threat_model, num_samples=n_sample)

    attack_summary = threat_model.test(attacker, num_samples = n_sample)
    metrics = attack_summary.get_metrics()

    return metrics