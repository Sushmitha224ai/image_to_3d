# DATASET AUDIT & PROVENANCE REPORT

```text
Dataset name                 : CUHK Face Sketch Database (CUFS) & CUHK Face Sketch FERET Database (CUFSF)
Official/source URL          : http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html & https://github.com/chaofengc/Face-Sketch-SCG
Dataset type                 : Genuine Artist-Drawn Pencil Sketches Paired with Real Face Photographs
Number of identities         : 1592
Number of sketch-photo pairs : 1592
Sketch creation method       : Viewed / Artist Sketch (Hand-drawn by professional sketch artists viewing frontal photos)
Photo source                 : Real human facial photographs (CUHK Students, AR Database, XM2VTS Database, FERET Database)
License/access restrictions  : Restricted to academic, educational, and non-commercial research use
Training identities          : 1114 (70%)
Validation identities        : 238 (15%)
Testing identities           : 240 (15%)
Identity leakage             : ZERO (Strict identity-based partition; no identity appears in more than one split)
Synthetic images present     : NO (All synthetic images moved to data/synthetic_backup/)
Final dataset used           : Genuine CUFS / CUFSF Paired Research Dataset (1592 pairs)
```

## SKETCH TAXONOMY DEFINITIONS
- **Viewed / Artist Sketch**: Hand-drawn pencil sketch created by a sketch artist directly viewing a frontal face photograph (used in this project dataset).
- **Semi-Forensic Sketch**: Hand-drawn sketch created based on artist memory or delayed recall of a subject.
- **Forensic Sketch**: Hand-drawn sketch created by a forensic sketch artist based solely on an eyewitness verbal description.
- **Synthetic Sketch**: Computer-generated drawing created by applying edge-detection algorithms or procedural filters to a photo.

## DATASET SUBSETS INCLUDED
1. **CUHK Student Dataset**: 188 identities (188 sketch-photo pairs)
2. **AR Database**: 123 identities (123 sketch-photo pairs)
3. **XM2VTS Database**: 295 identities (295 sketch-photo pairs)
4. **CUFSF Database (FERET)**: 986 identities (986 sketch-photo pairs)

Total Verified Usable Pairs: **1592**
