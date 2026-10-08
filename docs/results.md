# Results

Headline numbers written by `uv run gacha-risk experiment <name>`. Each `##` section is owned by one
experiment and is replaced when it reruns.

## e01_reproduce_paper

- E0: 62.3376639
- V0: 592.4344706
- hard_only_E0: 69.69980378
- T10_mean: 623.376639
- T10_q90: 719
- featured_mean: 93.50649585

## e02_endfield_single

- full_mean: 74.331213
- full_sd: 36.00526277
- full_q90: 115
- full_support_max: 115
- two_copies_mean: 162.7694433
- two_copies_p_success: 1

## e03_personal_state

- E_fresh: 79.29137268
- E_t79_n0: 40.46441446
- E_t0_n100: 19.25793385
- monotone_in_t0: 1

## e04_carry_over

- gap_K2: 0.2009439347
- gap_K3: 0.1519991124
- exact_mean_K2: 140.9465908
- iid_mean_K2: 148.662426
- exact_q90_K3: 277
- iid_q90_K3: 297

## e05_tail_risk

- q90_K1: 115
- q90_K3: 277
- q90_K5: 436
- mean_K5: 340.8224605
- cvar95_K5: 488.1170499
- completion_60_per_banner_K5: 0.2997557341

## e06_mc_convergence

- exact_survival_90: 0.3870302954
- mc_200k: 0.38755
- max_abs_z: 0.677009382

## e07_sensitivity

- mean_up_share_0.5: 74.331213
- mean_up_share_0.7: 62.02705958
- mean_no_ramp: 77.78707822
- mean_guarantee_100: 67.34311718
- mean_guarantee_140: 80.33987017
- mean_guarantee_none: 93.87579868
