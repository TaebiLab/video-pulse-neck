# Contactless Heart Rate and HRV Estimation from Neck Videos
This repository contains the implementation code for the paper:

**Contactless Heart Rate and Heart Rate Variability Estimation from Neck Videos**  
*Mohammad Muntasir Rahman, Amirtahà Taebi*  
Presented at the 2025 IEEE EMBC which was hold in Copenhagen, Denmark
[[Paper Link]](https://doi.org/10.XXXX/XXXXXXXX) *(link will be updated once available)*


## Overview
This project introduces a non-contact, video-based approach for estimating heart rate (HR) and heart rate variability (HRV) from short videos of the human neck. We evaluated six widely-used video-based pulse extraction methods:

- **GREEN**
- **CHROM** (Chrominance-based)
- **POS** (Plane-Orthogonal-to-Skin)
- **OMIT** (Orthogonal Matrix Image Transformation)
- **ICA** (Independent Component Analysis)
- **LGI** (Local Group Invariance)

All extracted signals were validated against **synchronized ECG** recordings. We also developed a robust, adaptive peak detection algorithm for estimating HR and HRV in noisy video-based signals.


## Contents



## Requirements

- MATLAB (R2022a or later)
- Python

Dependencies adapted from:
- 📦 [iPhys Toolbox](https://github.com/mcdufflab/iPhys)
- 📦 [rPPG-Toolbox](https://github.com/zhaoxiangyi0727/rPPG-Toolbox)

## Acknowledgments
This work was supported by:

- **National Science Foundation (NSF)** — Grant No. 2340020  
- **SMART Business Act Grant** — Grant No. 2024-04  
  *(Mississippi Institutions of Higher Learning)*

## Citation
If you use this code in your work, please cite:

```bibtex
@inproceedings{rahman2025contactless,
  title={Contactless Heart Rate and Heart Rate Variability Estimation from Neck Videos},
  author={Rahman, Mohammad Muntasir and Taebi, Amirtahà},
  booktitle={Proceedings of the IEEE EMBC 2025},
  year={2025}
}
