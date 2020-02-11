# Experiments

### DQN

|No  | Agent      | Parent | Timestamp     | Reward Function   | Avg Test Score | StdDev   | Change           | Comments |
|----|------------|--------|---------------|-------------------|----------------|----------|------------------|----------|
| 1  | agn_dqn_1  | -      | 200207 044139 | rf_info2d_pos     | 129.30         | ± 35.89  | -                |          |
| 2  | agn_dqn_2  | 1      | 200207 221202 | rf_info2d_pos     | 116.61         | ± 48.83  | t1:1200->2000    |          |
| 3  | agn_dqn_3  | 1      | 200208 005040 | rf_info2d_pos     | 54.00          | ± 63.31  | gamma:0.99->0.999|          |
| 4  | agn_dqn_4  | 1      | 200208 105916 | rf_info2d_pos     | 129.74         | ± 34.76  | gamma:0.99->0.98 |          |
| 5  | agn_dqn_5  | 1      | 200208 163200 | rf_spar_pos       | 74.70          | ± 103.04 | rf               |          |
| 6  | agn_dqn_6  | 1      | 200209 000400 | None              | 5.86           | ± 9.16   | rf + InNoise=0.5 | run5 ok  |
| 7  | agn_dqn_7  | 6      | 200209 100600 | None              | 17.40          | ± 75.47  | lr:1e-4->5e-4    |          |
| 8  | agn_dqn_8  | 6      | 200209 204454 | None              | -0.8           | ± 0.4    | gamma:0.99->1    |          |
| 9  | agn_dqn_9  | 6      | 200210 054024 | None              |  21.22         | ± 55.98  | InNoise=360      |          |
| 10 | agn_dqn_10 | 4      | 200210 054117 | rf_info2d_shp_pos | 49.66          | ± 35.18  | rf + InNoise=360 |          |
| 11 | agn_dqn_11 | 10     | 200210 114300 | rf_info2d_shp_pos | 22.11          | ± 26.07  | InNoise->None    |          |
| 12 | agn_dqn_12 | 1      | 200210 181100 | rf_info2d_pos     |  7.71          | ± 9.35   | InNoise->None    |          |
| 13 | agn_dqn_13 | 1      | 200210 230800 | rf_info2d_pos     |  44.16         | ± 23.64  | rb:True->False   |          |
| 14 | agn_dqn_14 | 1      | 200211 015000 | rf_info2d_pos     | 100.50         | ± 55.36  | edecay:exp->lin  |          |
| 15 | agn_dqn_15 | 1      | 200211 015000 | rf_info2d_pos     |          | ±   | edecay:no annealing  |          |

### TD3

|No  | Agent      | Parent | Timestamp     | Reward Function   | Avg Test Score | StdDev   | Change           | Comments |
|----|------------|--------|---------------|-------------------|----------------|----------|------------------|----------|
| 1  | agn_td3_1  | -      | 200210 024752 | rf_info2d_pos     |  20.44         | ± 25.14  | -                | run3 ok  |
| 2  | agn_td3_2  | 1      | 200210 114100 | rf_info2d_shp_pos |  0.0019        | ± 0.0010 | rf               |          |
| 3  | agn_td3_3  | 1      | 200210 180400 | rf_info2d_pos     |  2.71          | ± 0.13   | tau:0.01->0.1    |          |
| 4  | agn_td3_4  | 1      | 200210 191900 | rf_info2d_pos     |  19.84         | ± 34.24  | actnoise:0.2->0.1|          |
| 5  | agn_td3_5  | 1      | 200210 231000 | rf_info2d_pos     |  38.30         | ± 32.35  | InNoise=360      |          |

## Reinforce

|No  | Agent      | Parent | Timestamp     | Reward Function   | Avg Test Score | StdDev   | Change           | Comments |
|----|------------|--------|---------------|-------------------|----------------|----------|------------------|----------|
| 1  | agn_rfc_1  | -      | 200210 120500 | rf_info2d_shp_pos |  0.0005        | ± 0.0007 | -                |          |
| 2  | agn_rfc_2  | 2      | 200210 180600 | rf_info2d_pos     |    .44         | ±        | rf               |          |
| 3  | agn_rfc_3  | 1      | 200210 00 | rf_info2d_pos     |    .44         | ±        | rf               |          |