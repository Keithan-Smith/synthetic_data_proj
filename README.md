# Synthetic Retail PD/LGD — Hybrid Generator + Simulator + One-click Report

## Run (German Credit)

1) Put UCI German Credit CSV at: /mnt/data/german_credit.csv
2) Run:
   python /mnt/data/synth_project/pipelines/cli.py --config /mnt/data/synth_project/configs/german_example.yaml
3) Open the HTML report:
   /mnt/data/synth_project/reports/german_run/index.html

## Run (Prior mode — no dataset)

   python /mnt/data/synth_project/pipelines/cli.py --config /mnt/data/synth_project/configs/prior_mode.yaml

## Profiles (tuned defaults)
- fast:     VAE=3,  GAN=5,   batch=128
- balanced: VAE=10, GAN=30,  batch=256   (default; CPU-friendly)
- thorough: VAE=30, GAN=100, batch=512   (use if you have a strong machine/GPU)

## PD hazard knobs
- pd_logit_shift: add to logit(base monthly hazard) to raise/lower default rates globally
- macro_pd_mult:  sensitivity of hazard to unemployment deviations from 5%

## Privacy
Set privacy_enabled: true and choose dp_noise_multiplier to target an epsilon.

