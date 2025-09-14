import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

def run_ctgan_baseline(df, n_samples=None):
    """Return (status_dict, synth_df|None). No-op if SDV not installed."""

    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df)

    model = CTGANSynthesizer(meta, epochs=300)
    model.fit(df)
    n = n_samples or len(df)
    synth = model.sample(n)
    return {"status": "ok"}, synth
