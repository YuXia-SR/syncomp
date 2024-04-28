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

def string_conv_int(x):
    mapping = {v: i for i, v in enumerate(set(x))}
    return np.array(list(map(mapping.__getitem__, x)))

def distance_closest_record(real_df, syn_df, n_sample=1000, random_state=0):
    def encode_integer(df):
        string_columns = df.select_dtypes(include=['object', 'bool']).columns

        # Encode string columns as integers
        for col in string_columns:
            df[col] = string_conv_int(df[col])

        return df

    real_df = real_df.sample(n_sample, random_state=random_state).fillna(0)
    syn_df = syn_df.fillna(0)

    real_df = encode_integer(real_df)
    syn_df = encode_integer(syn_df)

    def compute_distance(df, row):
        distance = np.linalg.norm(df - row, axis=1)
        return distance.min()
    
    dcr = real_df.apply(lambda x: compute_distance(syn_df, x), axis=1)

    return dcr.min()

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
    
    with open("../results/complete_dataset_filtered.json") as f:
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
    attacker = tapas.attacks.GroundhogAttack()
    attacker = tapas.attacks.ShadowModellingAttack(
    tapas.attacks.FeatureBasedSetClassifier(
        tapas.attacks.NaiveSetFeature() + tapas.attacks.HistSetFeature() + tapas.attacks.CorrSetFeature(),
        classifier
    ),
    label = "Groundhog"
    )
    attacker.train(threat_model, num_samples=n_sample)

    attack_summary = threat_model.test(attacker, num_samples = n_sample)
    metrics = attack_summary.get_metrics()

    return metrics