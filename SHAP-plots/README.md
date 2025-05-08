# Shapley Value Analysis for Spoofed Speech Attribution

This directory presents the Shapley value visualizations for our paper:

> **"Towards Explainable Spoofed Speech Attribution and Detection: a Probabilistic Approach for Characterizing Speech Synthesizer Components"**
> *Jagabandhu Mishra, Manasi Chhibber, Hye-jin Shim, Tomi H. Kinnunen*
> \[Computer Speech & Language, 2025]

## SHAP-Plots

The SHAP (SHapley Additive exPlanations) plots in this folder illustrate the **relative contribution of each attribute-value** in the **spoofing detection** and **spoofing attack attribution** tasks. The attributes correspond to interpretable components of the speech synthesis or voice conversion pipelines, such as:

* `inputs`: input modality (e.g., text, human speech, TTS-generated speech)
* `input processor`: NLP, WORLD, LPCC/MFCC, etc.
* `duration model`: HMM, FF, attention, etc.
* `conversion model`: AR-RNN, FF, GMM-UBM, etc.
* `speaker representation`: VAE, one-hot, d-vector, etc.
* `outputs`: acoustic features like MFCC-F0, LPC
* `waveform generator`: WaveNet, WORLD, WaveRNN, etc.

Each plot corresponds to either:

* A binary classification case (e.g., `bonafide` vs. `spoof`).
* Or an individual spoofing attack type (`A01` to `A18`) from ASVspoof 2019.

The height of the bars reflects the **average absolute Shapley value**, indicating how influential a particular attribute-value was in the model’s prediction.

---

## Key Findings from the Plots

### Spoofing Detection (`bonafide` vs. `spoof`)

* The **`waveform generator`**, particularly `WaveNet` and `WORLD`, shows the strongest influence in classifying bonafide vs. spoofed speech.
* **Inputs** (text vs. speech) and **conversion-related components** (e.g., AR-RNN) also contribute significantly to classification.
* For **spoofed** speech, lower Shapley values across most attributes suggest a more distributed or ambiguous influence pattern—possibly due to diverse spoofing strategies.

### Attack Attribution (`A01`–`A18`)

* Different attacks rely on **distinct combinations of attributes**, consistent with their underlying synthesis/VC architectures.
* Despite overlaps, certain attributes (like `waveform generator`) remain **consistently dominant** across many attacks, making them strong indicators for model interpretability and attribution.

## Reference

If you use these results or visualizations, please cite our paper:

```
@article{mishra2025spoof,
  title={Towards Explainable Spoofed Speech Attribution and Detection: a Probabilistic Approach for Characterizing Speech Synthesizer Components},
  author={Mishra, Jagabandhu and Chhibber, Manasi and Shim, Hye-jin and Kinnunen, Tomi},
  journal={Computer Speech and Language},
  year={2025}
}
```
