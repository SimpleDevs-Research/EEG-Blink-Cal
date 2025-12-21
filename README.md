# EEG-Blink-Cal

**Derived from:** [eeg_blink_calibration](https://github.com/SimpleDevs-Research/eeg_blink_calibration)

## Initializing a Python Environment

**Expected Python Version**: `3.11

<details>
<summary><strong>Mac OSX</strong></summary>

```bash
pyenv global 3.11           # Switch to necessary Python version via `pyenv`
python3.11 -m venv .venv    # Create `.venv`
source .venv/bin/activate   # Activate environment
# --- Do whatever you need to do ---
deactivate                  # Deactivate environment
```
</details>

<details>
<summary><strong>Windows</strong></summary>

```bash
py -3.11 -m venv .venv      # Create `.venv`
.venv/Scripts/activate.ps1  # Activate environment
# --- Do whatever you need to do ---
deactivate                  # Deactivate environment
```
</details>

## Installing Dependencies

```bash
pip install -r requirements.txt
```

## IMU Calibration

Data comes from comparisons between IMU data from the **Muse 2** EEG headband and simulated IMU data from a Meta Quest Pro in Unity. The command to run the necessary analysis is the following:

```bash
# Template Command
python imu_calibration.py <ROOT/DiRECTORY> <VR/IMU/FILENAME> <VR/IMU/COLNAME> <EEG/IMU/FILENAME> <EEG/IMU/COLNAME> -sb <MILLISECS> -eb <MILLISECS>

# Example
python imu_calibration.py ./samples/imu_calibration/ head_imu.csv gyro_x eeg.csv gyro_y -sb 3000 -eb 100
```

### Output Files

1. `offsets.csv` = estimated offsets aggregated from all trials
2. `<EXPERIMENTT NAME>_offsets.png` = the figure depicting all offsets aggregated from all trials

### Tests for Significance

We classified our variables based on the following:

|Variable|Description|Type|Values|
|:-|:-|:-|:-|
|`conf_threshold`|The confidence threshold setting for eye tracking in Meta SDK's `OVREyeGaze` component.|Continuous|Numeric|
|`fps`|The designated FPS of the simulation session.|Discrete numeric|Numeric|
|`vr_status`|Whether the VR device was turned off, quit, or left on between trials.|Categorical|"off" / "on" / "quit"|
|`eeg_status`|Whether the Muse device was turned off or left on between trials.|Categorical|"off" / "on"|
|`distance`|Whether the Mind Monitor device was withina meter or two meters away from the Muse device.|Categorical| "Near" / "Far"|

This is a mix of continuous and categorical predictors. This rules out the following:

- plain Pearson correlation
- Pairwise tests due to confounding variables

Instead, we employ two strategies:

#### First step: Multiple Linear Regression (OLS)

This step is primarily to controlf for confounders between potential factors of offset values.

$$
\text{offset}_{i} = \beta_0 + \beta_1 c_i + \beta_2 f_i + \beta_3 v_i + \beta_4 e_i + \beta_5 d_i + \epsilon_i
$$

where:

- $c$: Confidence Threshold
- $f$: Sample Rate / Frequency / FPS
- $v$: VR status between trials
- $e$: EEG status between trials
- $d$: Distance between Mind Monitor and Muse device

```
                             OLS Regression Results
================================================================================
Dep. Variable:     Q("offset_eeg-gaze")   R-squared:                       0.825
Model:                              OLS   Adj. R-squared:                  0.820
Method:                   Least Squares   F-statistic:                     188.1
Date:                  Sun, 21 Dec 2025   Prob (F-statistic):          6.37e-102
Time:                          01:59:25   Log-Likelihood:                -1750.4
No. Observations:                   288   AIC:                             3517.
Df Residuals:                       280   BIC:                             3546.
Df Model:                             7
Covariance Type:              nonrobust
==========================================================================================
                             coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------------
Intercept               -156.9271     16.686     -9.405      0.000    -189.772    -124.082
conf_threshold[T.0.5]      6.8333     15.448      0.442      0.659     -23.576      37.242
conf_threshold[T.0.75]    21.9479     15.448      1.421      0.156      -8.461      52.357
fps[T.90]                 15.0000     12.613      1.189      0.235      -9.829      39.829
C(vr_status)[T.on]      -110.1944     18.566     -5.935      0.000    -146.741     -73.648
C(vr_status)[T.quit]    -532.7083     17.838    -29.864      0.000    -567.821    -497.595
C(eeg_status)[T.on]      452.8889     18.566     24.393      0.000     416.342     489.436
C(distance)[T.far]        36.0417     15.448      2.333      0.020       5.633      66.451
==============================================================================
Omnibus:                        5.761   Durbin-Watson:                   0.378
Prob(Omnibus):                  0.056   Jarque-Bera (JB):                6.706
Skew:                           0.189   Prob(JB):                       0.0350
Kurtosis:                       3.645   Cond. No.                         6.08
==============================================================================
```

- `conf_threshold` is NOT statistically significant. ($p_{c=0.5} = 0.659$, $p_{c=0.75} = 0.156$). Once other variables are controlled for, confidence threshold does not meaningfully affect EEG–gaze offset.
- `fps` is not statistically significant ($p_s = 0.235$). Frame rate differences (72 vs 90) do not significantly affect offset in this dataset.
- `vr_status` is **statistically significant** ($p_v \lt 0.001$). VR state between trials massively shifts timing alignment.
- `eeg_status` is **statistically significant** ($p_e \lt 0.001$). When EEG is ON (vs OFF), offsets increase by ~450 ms, controlling for everything else. This may be due to:
    - buffering
    - synchronization delays
    - pipeline changes
- `distance` is **weakly statistically significant** ($p_d = 0.02$). Far distance slightly increases offset relative to near distance.
- **Durban-Watson** statistic $= 0.379$ suggests positive autocorrelation, likely due to repeated trials. This means that errors are not independent.

> Most of the EEG–gaze timing offset is driven by system state, not experimental tuning parameters. Specifically:
> - Turning EEG on/off and VR on/quit introduces hundreds of milliseconds
> - Distance adds a small but consistent effect
> - Frame rate and confidence threshold do not materially influence offset.

#### Second Step: Mixed-Effect Models

After accounting for systematic differences between sessions/subjects/runs (the groups), which factors still explain variation in EEG–gaze offset?

```
                 Mixed Linear Model Regression Results
========================================================================
Model:              MixedLM   Dependent Variable:   Q("offset_eeg-gaze")
No. Observations:   288       Method:               REML
No. Groups:         12        Scale:                1764.6750
Min. group size:    24        Log-Likelihood:       -1479.4337
Max. group size:    24        Converged:            Yes
Mean group size:    24.0
------------------------------------------------------------------------
                         Coef.   Std.Err.   z    P>|z|  [0.025   0.975]
Intercept               -156.927   73.659 -2.130 0.033 -301.297  -12.557
conf_threshold[T.0.5]      6.833    6.063  1.127 0.260   -5.051   18.717
conf_threshold[T.0.75]    21.948    6.063  3.620 0.000   10.064   33.832
fps[T.90]                 15.000    4.951  3.030 0.002    5.297   24.703
C(vr_status)[T.on]      -110.194  108.240 -1.018 0.309 -322.341  101.952
C(vr_status)[T.quit]    -532.708  103.994 -5.123 0.000 -736.532 -328.885
C(eeg_status)[T.on]      452.889  108.240  4.184 0.000  240.742  665.035
C(distance)[T.far]        36.042   90.061  0.400 0.689 -140.475  212.558
Group Var              16148.470  209.023
========================================================================
```

- `conf_threshold` ($p_{c=0.5} = 0.260$, $p_{c=0.75} \lt 0.001$): Moving from baseline → 0.75 increases offset by ~22 ms. However, 0.5 does not differ reliably from baseline. Basically, only higher confidence thresholds systematically increase offset.
- `fps` ($p_s = 0.002$): Running at 90 fps increases offset by ~15 ms vs baseline fps. In other words, FPS is a real contributor, not just a confound.
- `vr` ($p_{v=\text{on}} = 0.309$, $p_{v=\text{quit}} \lt 0.001$): "vr=quit" has a very large and highly reliable effect, while "vr=on" HAD significance in OLS but not here. **VR quitting fundamentally alters offset**.
    - The "vr=on" effect was largely explained by group differences.
    - "vr=quit" is so large that it remains significant even after accounting for groups.
- `eeg_status` ($p_e \lt 0.001$): EEG being “on” shifts offset by ~450 ms. **EEG state is a dominant driver of offset.**
- `distance` ($p_d = 0.689): Distance **lost its significance**. The distance effect in OLS was likely due to between-group structure.
- Random Effects: $\text{Group Var} = 16148.47$.
    - Very huge residual variance. AKA there are large baseline differences between groups. OLS (which ignores groupings) inflates things a lot.

#### Final Statistical Significance Results

|Factor|OLS|Mixe|Verdict|
|:-|:-|:-|:-|
|`conf_threshold = 0.75`| Y | Y |Robust|
|`fps`| - | Y |  Hidden by confounding|
|`vr_status = on`| Y | - | Group artifact|
|`vr_status = quit`| Y | Y |Very robust|
|`eeg_status = on`| Y | Y | Strongest effect|
|`distance`| Y | - |Not robust|

Statistically supported contributors to offset:

- EEG status (on/off) — dominant effect
- VR quit state — massive negative shift
- High confidence threshold (0.75) — moderate positive effect
- Higher FPS (90) — small but reliable effect

Factors not independently predictive:
- VR simply being "on"
- Distance

#### Multicollinearity Check

https://www.statsmodels.org/dev/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html

General rule of thumb:

- VIF < 5 → fine
- VIF 5–10 → caution
- VIF > 10 → serious collinearity

```
                 variable       VIF
0               Intercept  7.000000
1   conf_threshold[T.0.5]  1.333333
2  conf_threshold[T.0.75]  1.333333
3               fps[T.90]  1.000000
4      C(vr_status)[T.on]  1.625000
5    C(vr_status)[T.quit]  1.500000
6     C(eeg_status)[T.on]  2.166667
7      C(distance)[T.far]  1.125000
```

We are generally fine - there is no evidence that the large eeg_status coefficient is an artifact of multicollinearity or simple sample-count bias.