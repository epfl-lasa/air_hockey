# Data Folder

Find here the recorder data for airhockey, organized on different folders :

- airhockey : raw recorded data
- airhockey_processed : datasets processed in one file
- *_flux_datasets : copies of the raw data organized in datasets
- figures used in the paper
- airhockey_ekf : reformatted raw_data to be used for the MATLAB EKF



## Datasets

D1 - Object 1, config 1, flat side, Fixed box pos, random flux
2000 hits clean, 600 hits dirty

D2 -  Object 2, config 1, flat side, Fixed box pos, random flux
2000 hits clean, 900 hits dirty

D3 -  Object 1, config 2, flat side, Fixed box pos, random flux
300 hits clean, 150 hits dirty(?)

D4 -  Object 2, config 2, flat side, Fixed box pos, random flux
300 hits clean

D5 - Object 3 (cylinder), Config 1, fixed box pos, (currently fixed flux)
120 hits


D*_dirty -> paper surface attached underneath the object is dirty (dust mainly), whichs greatly increases friction
D*_clean -> paper surface attached underneath the object is clean/new, whichs greatly reduces friction


D1-edge - Object 1, config 1, edge side, Fixed box pos, random flux 
	-> reduces variance by a lot-> variance due a lot to orientation of box at hit
300 hits

D1-fixed flux - Object 1, config 1, flat side, random pos, fix flux
-> best robot interaction, film videos with this 
120 hits

DA-Inertia-consistency- 50 hits for each obj and each config, Fixed box pos, fixed flux, NS Inertia shaping
200 hits total

D2-Inertia - Object 2, config 1, flat side, Fixed box pos, random flux, NS Inertia shaping
300 hits

D1-Inertia - Object 1, config 1, flat side, Fixed box pos, random flux, NS Inertia shaping
100 hits

D1-Inertia_reduced - Object 1, config 1, flat side, Fixed box pos, random flux, NS Inertia shaping at 10%
